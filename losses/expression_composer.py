import ast
from omegaconf import ListConfig
import torch
import torch.nn as nn


class _IntermediateExtractor(ast.NodeVisitor):
    """Visitor AST to extract intermediate expressions"""
    def __init__(self):
        self.intermediates = []

    def visit_BinOp(self, node):
        # detect "A @ x"
        if isinstance(node.op, ast.MatMult):
            left = self._to_str(node.left)
            right = self._to_str(node.right)
            expr = f"{left}@{right}"
            self.intermediates.append(expr)
        self.generic_visit(node)

    def visit_Call(self, node):
        # detect "AE(x)", "A(B(x))", ...
        fname = self._to_str(node.func)
        args = ",".join([self._to_str(a) for a in node.args])
        expr = f"{fname}({args})"
        self.intermediates.append(expr)
        self.generic_visit(node)

    def _to_str(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return node.attr
        elif isinstance(node, ast.Call):
            # nested call A(B(x))
            fname = self._to_str(node.func)
            args = ",".join([self._to_str(a) for a in node.args])
            return f"{fname}({args})"
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            # nested matmul
            left = self._to_str(node.left)
            right = self._to_str(node.right)
            return f"{left}@{right}"
        else:
            return "unknown"


class _CompiledTerm(nn.Module):
    def __init__(self, name, fn, weight, code, required_vars, intermediate_exprs):
        """Small struct-like holder for compiled single loss term."""
        super().__init__()
        self.name = name
        self.fn = fn
        self.weight = weight
        self.code = code
        self.required_vars = required_vars
        self.intermediate_exprs = intermediate_exprs


class ExpressionLossComposer(nn.Module):
    """
    Flexible expression-based loss composer.
    Supports:
        - Python expressions: mse(y, L @ x), l1(x, AE(x)), ...
        - Auto-detected variables & intermediates via AST
        - Auto-created trainable weights
        - External variables injected by Hydra
    """

    def __init__(self, terms, **external_vars):
        """
        Args:
            terms: list of dicts describing each loss term.
            external_vars: variables/modules/tensors passed from Hydra config.
                           These become permanent part of the composer context.
                           Example: AE=${model.ae}, L=${leadfield}
        """
        super().__init__()

        if not isinstance(terms, (list, tuple, ListConfig)):
            raise TypeError("terms must be list of dicts.")
        
        self.external_vars = {}
        for k, v in external_vars.items():
            if isinstance(v, torch.Tensor):
                # This ensures Lightning will move it to CUDA
                self.register_buffer(k, v, persistent=False)
            else:
                # Non-tensor (nn.Module, int, float, string) – leave as is
                self.external_vars[k] = v

        self._terms = nn.ModuleList()   # store terms in a structured list

        for cfg in terms:
            name = cfg.get("name", "unnamed")
            fn = cfg["fn"]
            expr = cfg["expr"]
            weight = cfg.get("weight", None)

            # Compile expression and detect variables (AST)
            parsed = ast.parse(expr, mode="eval")

            auto_vars = [
                node.id
                for node in ast.walk(parsed)
                if isinstance(node, ast.Name) and node.id not in ("fn",)
            ]
            required_vars = list(dict.fromkeys(auto_vars))  # unique & ordered

            # Extract intermediate (L@x, AE(x), A(B(x)))
            i_ex = _IntermediateExtractor()
            i_ex.visit(parsed)
            intermediate_exprs = i_ex.intermediates  # list[str]
            
            code = compile(parsed, "<expr>", "eval")

            # Build weight (trainable if user does not specify one)
            if weight is None:
                weight = nn.Parameter(torch.tensor(1.0))
                self.register_parameter(f"{name}_weight", weight)
            elif isinstance(weight, (float, int)):
                weight = torch.tensor(float(weight))
            # if weight is already Parameter or nn.Module, accept it

            # Store term
            self._terms.append(
                _CompiledTerm(
                    name=name,
                    fn=fn,
                    weight=weight,
                    code=code,
                    required_vars=required_vars,
                    intermediate_exprs=intermediate_exprs
                )
            )

    # -------------------------------------------------------------------------
    def _build_context(self, **kwargs):
        """
        Create execution context for eval() of each term.
        Context priority:
            kwargs > external_vars
        """
        context = {}
        context.update(self.external_vars)
        context.update(self._buffers)
        context.update(kwargs)
        return context
    
    # Evaluate 1 intermediate expression safely
    def _eval_intermediate(self, expr: str, ctx: dict):
        """
        expr: "L@x", "AE(x)", "A(B(x))"
        -> compile and eval like eval(expr, ctx)
        """
        try:
            code = compile(expr, "<intermediate>", "eval")
            return eval(code, {"__builtins__": {}}, ctx)
        except Exception:
            return None

    # -------------------------------------------------------------------------
    def compute_raw_terms(self, return_intermediate=False, **kwargs):
        """Return raw (unweighted) term + intermediates values as dict(name -> Tensor)."""
        ctx = self._build_context(**kwargs)
        raw = {}
        inter = {"input": {}}  # input-level variables
        
        if return_intermediate:
            for k, v in ctx.items():
                if isinstance(v, (torch.Tensor)): #isinstance(v, (torch.Tensor, nn.Module))
                    inter["input"][k] = v

        for term in self._terms:
            # compute raw loss
            local_ctx = dict(ctx)
            local_ctx["fn"] = term.fn
            val = eval(term.code, {"__builtins__": {}}, local_ctx)
            raw[term.name] = val
            
            # compute intermediates for this term
            if return_intermediate:
                inter[term.name] = {}
                for expr in term.intermediate_exprs:
                    out = self._eval_intermediate(expr, local_ctx)
                    if isinstance(out, (torch.Tensor)):
                        inter[term.name][expr] = out

        if return_intermediate:
            return raw, inter
        return raw

    # -------------------------------------------------------------------------
    def compute_weighted_terms(self, return_intermediate=False, **kwargs):
        """Return weighted term + intermediates values as dict(name -> Tensor)."""
        if return_intermediate:
            raw, inter = self.compute_raw_terms(True, **kwargs)
        else:
            raw = self.compute_raw_terms(False, **kwargs)
            inter = None

        weighted = {}
        for term in self._terms:
            w = term.weight
            if isinstance(w, torch.Tensor):
                weighted[term.name] = raw[term.name] * w
            else:
                weighted[term.name] = raw[term.name] * float(w)

        if return_intermediate:
            return weighted, inter
        return weighted

    # -------------------------------------------------------------------------
    def forward(self, return_terms=False, return_intermediate=False, **kwargs):
        """
        Compute total loss.
        """
        if return_intermediate:
            weighted, inter = self.compute_weighted_terms(True, **kwargs)
        else:
            weighted = self.compute_weighted_terms(False, **kwargs)
            inter = None

        total = None
        for v in weighted.values():
            total = v if total is None else total + v

        if return_terms and return_intermediate:
            return total, weighted, inter
        if return_terms:
            return total, weighted
        if return_intermediate:
            return total, inter
        return total

