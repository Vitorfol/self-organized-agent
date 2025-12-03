"""Validator to check if generated code is actually implemented."""

import ast
from typing import Tuple, List, Set


def has_pass_only(func_node: ast.FunctionDef) -> bool:
    """Check if a function only contains 'pass' statement (no real implementation)."""
    if len(func_node.body) == 0:
        return True
    
    body = func_node.body
    
    # Skip docstring if present
    start_idx = 0
    if (len(body) > 0 and 
        isinstance(body[0], ast.Expr) and 
        isinstance(body[0].value, (ast.Str, ast.Constant))):
        start_idx = 1
    
    # Check remaining statements
    remaining = body[start_idx:]
    
    if len(remaining) == 0:
        return True
    
    if len(remaining) == 1 and isinstance(remaining[0], ast.Pass):
        return True
    
    # Check if it's only "..." (Ellipsis)
    if len(remaining) == 1 and isinstance(remaining[0], ast.Expr):
        if isinstance(remaining[0].value, (ast.Constant, ast.Ellipsis)):
            return True
    
    return False


def extract_function_calls(code: str) -> Set[str]:
    """Extract all function calls from code."""
    try:
        tree = ast.parse(code)
        calls = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
        
        return calls
    except SyntaxError:
        return set()


def extract_function_definitions(code: str) -> Set[str]:
    """Extract all function definitions from code."""
    try:
        tree = ast.parse(code)
        defs = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defs.add(node.name)
        
        return defs
    except SyntaxError:
        return set()


# Common Python built-in functions and methods
BUILTIN_FUNCTIONS = {
    'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set',
    'tuple', 'bool', 'type', 'isinstance', 'hasattr', 'getattr', 'setattr',
    'enumerate', 'zip', 'map', 'filter', 'sorted', 'sum', 'min', 'max',
    'abs', 'round', 'open', 'input', 'format', 'all', 'any', 'next',
    'iter', 'reversed', 'ord', 'chr', 'bin', 'hex', 'oct', 'pow',
    # Common string methods
    'join', 'split', 'strip', 'lower', 'upper', 'replace', 'startswith',
    'endswith', 'find', 'rfind', 'index', 'rindex', 'count', 'isdigit',
    'isalpha', 'isalnum', 'isspace', 'islower', 'isupper', 'ljust', 'rjust',
    'center', 'zfill', 'lstrip', 'rstrip',
    # Common list/dict methods
    'append', 'extend', 'insert', 'remove', 'pop', 'clear', 'copy',
    'get', 'keys', 'values', 'items', 'update', 'setdefault',
    # Common built-in exceptions
    'ValueError', 'TypeError', 'KeyError', 'IndexError', 'AttributeError',
}


def validate_implementation(code: str, strict: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate that generated code is actually implemented.
    
    Args:
        code: The Python code to validate
        strict: If True, also check for undefined function calls
        
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, [f"Syntax error: {e}"]
    
    # Check for unimplemented functions (only pass/... statements)
    unimplemented = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if has_pass_only(node):
                unimplemented.append(node.name)
    
    if unimplemented:
        issues.append(f"❌ Functions not implemented (only 'pass' or '...'): {', '.join(unimplemented)}")
    
    # Check for undefined function calls (optional, can be noisy)
    if strict:
        calls = extract_function_calls(code)
        defs = extract_function_definitions(code)
        
        undefined = calls - defs - BUILTIN_FUNCTIONS
        
        if undefined:
            issues.append(f"⚠️  Code calls undefined functions: {', '.join(sorted(undefined))}")
    
    return len(issues) == 0, issues


def is_code_empty(code: str) -> bool:
    """Check if code is empty or contains no real implementation."""
    if not code or not code.strip():
        return True
    
    try:
        tree = ast.parse(code)
        # Check if there are any function definitions
        has_functions = any(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
        return not has_functions
    except SyntaxError:
        return True


if __name__ == "__main__":
    # Test with the actual problematic code
    test_code = '''
def create_a_entity(*args, **kwargs):
    """create a entity student with email field"""
    entity = initialize_entity(args, kwargs)
    entity_with_email = add_email_field(entity)
    return entity_with_email

def initialize_entity(args, kwargs):
    """Initialize an entity."""
    pass

def add_email_field(entity):
    """Add email field."""
    pass
'''
    
    is_valid, issues = validate_implementation(test_code, strict=False)
    print(f"Is valid: {is_valid}")
    for issue in issues:
        print(f"  {issue}")
    
    # Test with good code
    good_code = '''
def create_student(name, email):
    """Create a student entity."""
    return {"name": name, "email": email}
'''
    
    is_valid, issues = validate_implementation(good_code)
    print(f"\nGood code is valid: {is_valid}")
    for issue in issues:
        print(f"  {issue}")
