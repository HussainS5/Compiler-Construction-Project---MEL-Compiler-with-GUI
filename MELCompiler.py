"""
MEL (Mathematical Expression Language) Compiler with GUI (tkinter)
Complete implementation: Lexing, Parsing, Semantic Analysis,
Intermediate Code Generation, Interpretation + simple UI
"""

import re
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from enum import Enum
from typing import List, Any

# ==================== PHASE 1: LEXICAL ANALYSIS ====================

class TokenType(Enum):
    KEYWORD = "KEYWORD"
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    OPERATOR = "OPERATOR"
    DELIMITER = "DELIMITER"
    COMMENT = "COMMENT"
    EOF = "EOF"

class Token:
    def __init__(self, type: TokenType, value: Any):
        self.type = type
        self.value = value
    
    def __repr__(self):
        return f"Token({self.type.value}, {self.value})"

class Lexer:
    def __init__(self, code: str):
        self.code = code
        self.pos = 0
        self.current_char = self.code[self.pos] if self.code else None
        self.keywords = ['let', 'if', 'else', 'while', 'print', 'func', 'return']
    
    def advance(self):
        self.pos += 1
        self.current_char = self.code[self.pos] if self.pos < len(self.code) else None
    
    def skip_whitespace(self):
        while self.current_char and self.current_char.isspace():
            self.advance()
    
    def skip_comment(self):
        if self.current_char == '#':
            while self.current_char and self.current_char != '\n':
                self.advance()
    
    def number(self):
        num_str = ''
        dot_count = 0
        while self.current_char and (self.current_char.isdigit() or self.current_char == '.'):
            if self.current_char == '.':
                dot_count += 1
                if dot_count > 1:
                    raise SyntaxError("Invalid number format")
            num_str += self.current_char
            self.advance()
        return Token(TokenType.NUMBER, float(num_str) if '.' in num_str else int(num_str))
    
    def identifier(self):
        id_str = ''
        while self.current_char and (self.current_char.isalnum() or self.current_char == '_'):
            id_str += self.current_char
            self.advance()
        
        if id_str in self.keywords:
            return Token(TokenType.KEYWORD, id_str)
        return Token(TokenType.IDENTIFIER, id_str)
    
    def get_next_token(self):
        while self.current_char:
            if self.current_char.isspace():
                self.skip_whitespace()
                continue
            
            if self.current_char == '#':
                self.skip_comment()
                continue
            
            if self.current_char.isdigit():
                return self.number()
            
            if self.current_char.isalpha() or self.current_char == '_':
                return self.identifier()
            
            # Multi-char operators handling (==, !=, <=, >=)
            if self.current_char in '+-*/=<>!':
                op = self.current_char
                self.advance()
                if self.current_char == '=' and op in '=!<>':
                    op += self.current_char
                    self.advance()
                return Token(TokenType.OPERATOR, op)
            
            if self.current_char in '(){};,':
                delim = self.current_char
                self.advance()
                return Token(TokenType.DELIMITER, delim)
            
            raise SyntaxError(f"Invalid character: {self.current_char}")
        
        return Token(TokenType.EOF, None)
    
    def tokenize(self):
        tokens = []
        token = self.get_next_token()
        while token.type != TokenType.EOF:
            tokens.append(token)
            token = self.get_next_token()
        tokens.append(token)
        return tokens

# ==================== PHASE 2: SYNTAX ANALYSIS ====================

class ASTNode:
    pass

class Program(ASTNode):
    def __init__(self, statements):
        self.type = 'Program'
        self.body = statements

class Declaration(ASTNode):
    def __init__(self, id, value):
        self.type = 'Declaration'
        self.id = id
        self.value = value

class Assignment(ASTNode):
    def __init__(self, id, value):
        self.type = 'Assignment'
        self.id = id
        self.value = value

class IfStatement(ASTNode):
    def __init__(self, condition, consequent, alternate=None):
        self.type = 'IfStatement'
        self.condition = condition
        self.consequent = consequent
        self.alternate = alternate

class WhileStatement(ASTNode):
    def __init__(self, condition, body):
        self.type = 'WhileStatement'
        self.condition = condition
        self.body = body

class PrintStatement(ASTNode):
    def __init__(self, value):
        self.type = 'PrintStatement'
        self.value = value

class BinaryOp(ASTNode):
    def __init__(self, op, left, right):
        self.type = 'BinaryOp'
        self.op = op
        self.left = left
        self.right = right

class Number(ASTNode):
    def __init__(self, value):
        self.type = 'Number'
        self.value = value

class Identifier(ASTNode):
    def __init__(self, id):
        self.type = 'Identifier'
        self.id = id

class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        self.current_token = self.tokens[self.pos]
    
    def advance(self):
        self.pos += 1
        if self.pos < len(self.tokens):
            self.current_token = self.tokens[self.pos]
    
    def eat(self, token_type: TokenType, value=None):
        if self.current_token.type == token_type and (value is None or self.current_token.value == value):
            self.advance()
        else:
            expected = f"{token_type.value}{' '+value if value else ''}"
            raise SyntaxError(f"Expected {expected}, got {self.current_token.type.value} {self.current_token.value}")
    
    def parse(self):
        return self.program()
    
    def program(self):
        statements = []
        while self.current_token.type != TokenType.EOF:
            # skip stray semicolons
            if self.current_token.type == TokenType.DELIMITER and self.current_token.value == ';':
                self.advance()
                continue
            statements.append(self.statement())
        return Program(statements)
    
    def statement(self):
        if self.current_token.type == TokenType.KEYWORD:
            if self.current_token.value == 'let':
                return self.declaration()
            elif self.current_token.value == 'if':
                return self.if_statement()
            elif self.current_token.value == 'while':
                return self.while_statement()
            elif self.current_token.value == 'print':
                return self.print_statement()
        elif self.current_token.type == TokenType.IDENTIFIER:
            return self.assignment()
        
        raise SyntaxError(f"Unexpected token: {self.current_token}")
    
    def declaration(self):
        self.eat(TokenType.KEYWORD, 'let')
        id = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.OPERATOR, '=')
        value = self.expression()
        self.eat(TokenType.DELIMITER, ';')
        return Declaration(id, value)
    
    def assignment(self):
        id = self.current_token.value
        self.eat(TokenType.IDENTIFIER)
        self.eat(TokenType.OPERATOR, '=')
        value = self.expression()
        self.eat(TokenType.DELIMITER, ';')
        return Assignment(id, value)
    
    def if_statement(self):
        self.eat(TokenType.KEYWORD, 'if')
        condition = self.expression()
        self.eat(TokenType.DELIMITER, '{')
        consequent = []
        while not (self.current_token.type == TokenType.DELIMITER and self.current_token.value == '}'):
            consequent.append(self.statement())
        self.eat(TokenType.DELIMITER, '}')
        
        alternate = None
        if self.current_token.type == TokenType.KEYWORD and self.current_token.value == 'else':
            self.eat(TokenType.KEYWORD, 'else')
            self.eat(TokenType.DELIMITER, '{')
            alternate = []
            while not (self.current_token.type == TokenType.DELIMITER and self.current_token.value == '}'):
                alternate.append(self.statement())
            self.eat(TokenType.DELIMITER, '}')
        
        return IfStatement(condition, consequent, alternate)
    
    def while_statement(self):
        self.eat(TokenType.KEYWORD, 'while')
        condition = self.expression()
        self.eat(TokenType.DELIMITER, '{')
        body = []
        while not (self.current_token.type == TokenType.DELIMITER and self.current_token.value == '}'):
            body.append(self.statement())
        self.eat(TokenType.DELIMITER, '}')
        return WhileStatement(condition, body)
    
    def print_statement(self):
        self.eat(TokenType.KEYWORD, 'print')
        value = self.expression()
        self.eat(TokenType.DELIMITER, ';')
        return PrintStatement(value)
    
    def expression(self):
        return self.comparison()
    
    def comparison(self):
        node = self.additive()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.current_token.value in ['<', '>', '<=', '>=', '==', '!=']):
            op = self.current_token.value
            self.advance()
            node = BinaryOp(op, node, self.additive())
        
        return node
    
    def additive(self):
        node = self.multiplicative()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.current_token.value in ['+', '-']):
            op = self.current_token.value
            self.advance()
            node = BinaryOp(op, node, self.multiplicative())
        
        return node
    
    def multiplicative(self):
        node = self.primary()
        
        while (self.current_token.type == TokenType.OPERATOR and 
               self.current_token.value in ['*', '/']):
            op = self.current_token.value
            self.advance()
            node = BinaryOp(op, node, self.primary())
        
        return node
    
    def primary(self):
        if self.current_token.type == TokenType.NUMBER:
            value = self.current_token.value
            self.advance()
            return Number(value)
        elif self.current_token.type == TokenType.IDENTIFIER:
            id = self.current_token.value
            self.advance()
            return Identifier(id)
        elif self.current_token.type == TokenType.DELIMITER and self.current_token.value == '(':
            self.eat(TokenType.DELIMITER, '(')
            node = self.expression()
            self.eat(TokenType.DELIMITER, ')')
            return node
        
        raise SyntaxError(f"Unexpected token: {self.current_token}")

# ==================== PHASE 3: SEMANTIC ANALYSIS ====================

class SemanticAnalyzer:
    def __init__(self):
        self.symbol_table = {}
        self.errors = []
    
    def analyze(self, ast: ASTNode):
        self.visit(ast)
        return self.symbol_table, self.errors
    
    def visit(self, node):
        method_name = f'visit_{node.type}'
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)
    
    def generic_visit(self, node):
        raise Exception(f'No visit_{node.type} method')
    
    def visit_Program(self, node):
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_Declaration(self, node):
        if node.id in self.symbol_table:
            self.errors.append(f"Variable '{node.id}' already declared")
        else:
            self.symbol_table[node.id] = {'type': 'number', 'scope': 'global'}
        self.visit(node.value)
    
    def visit_Assignment(self, node):
        if node.id not in self.symbol_table:
            self.errors.append(f"Variable '{node.id}' not declared")
        self.visit(node.value)
    
    def visit_IfStatement(self, node):
        self.visit(node.condition)
        for stmt in node.consequent:
            self.visit(stmt)
        if node.alternate:
            for stmt in node.alternate:
                self.visit(stmt)
    
    def visit_WhileStatement(self, node):
        self.visit(node.condition)
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_PrintStatement(self, node):
        self.visit(node.value)
    
    def visit_BinaryOp(self, node):
        self.visit(node.left)
        self.visit(node.right)
    
    def visit_Number(self, node):
        pass
    
    def visit_Identifier(self, node):
        if node.id not in self.symbol_table:
            self.errors.append(f"Variable '{node.id}' not declared")

# ==================== PHASE 4: INTERMEDIATE CODE GENERATION ====================

class IntermediateCodeGenerator:
    def __init__(self):
        self.code = []
        self.temp_count = 0
        self.label_count = 0
    
    def new_temp(self):
        temp = f't{self.temp_count}'
        self.temp_count += 1
        return temp
    
    def new_label(self):
        label = f'L{self.label_count}'
        self.label_count += 1
        return label
    
    def emit(self, instruction):
        self.code.append(instruction)
    
    def generate(self, ast):
        self.visit(ast)
        return self.code
    
    def visit(self, node):
        method_name = f'visit_{node.type}'
        visitor = getattr(self, method_name, None)
        if visitor:
            return visitor(node)
        return None
    
    def visit_Program(self, node):
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_Declaration(self, node):
        value_temp = self.visit(node.value)
        self.emit(f'{node.id} = {value_temp}')
    
    def visit_Assignment(self, node):
        value_temp = self.visit(node.value)
        self.emit(f'{node.id} = {value_temp}')
    
    def visit_IfStatement(self, node):
        cond_temp = self.visit(node.condition)
        else_label = self.new_label()
        end_label = self.new_label()
        
        self.emit(f'if_false {cond_temp} goto {else_label}')
        for stmt in node.consequent:
            self.visit(stmt)
        self.emit(f'goto {end_label}')
        self.emit(f'{else_label}:')
        if node.alternate:
            for stmt in node.alternate:
                self.visit(stmt)
        self.emit(f'{end_label}:')
    
    def visit_WhileStatement(self, node):
        start_label = self.new_label()
        end_label = self.new_label()
        
        self.emit(f'{start_label}:')
        cond_temp = self.visit(node.condition)
        self.emit(f'if_false {cond_temp} goto {end_label}')
        for stmt in node.body:
            self.visit(stmt)
        self.emit(f'goto {start_label}')
        self.emit(f'{end_label}:')
    
    def visit_PrintStatement(self, node):
        value_temp = self.visit(node.value)
        self.emit(f'print {value_temp}')
    
    def visit_BinaryOp(self, node):
        left_temp = self.visit(node.left)
        right_temp = self.visit(node.right)
        result_temp = self.new_temp()
        self.emit(f'{result_temp} = {left_temp} {node.op} {right_temp}')
        return result_temp
    
    def visit_Number(self, node):
        return str(node.value)
    
    def visit_Identifier(self, node):
        return node.id

# ==================== PHASE 5 & 6: INTERPRETER (CODE GENERATION) ====================

class Interpreter:
    def __init__(self):
        self.variables = {}
        self.output = []
    
    def interpret(self, ast):
        self.visit(ast)
        return '\n'.join(map(str, self.output))
    
    def visit(self, node):
        method_name = f'visit_{node.type}'
        visitor = getattr(self, method_name, None)
        if visitor:
            return visitor(node)
        raise Exception(f'No visit_{node.type} method')
    
    def visit_Program(self, node):
        for stmt in node.body:
            self.visit(stmt)
    
    def visit_Declaration(self, node):
        value = self.visit(node.value)
        self.variables[node.id] = value
    
    def visit_Assignment(self, node):
        value = self.visit(node.value)
        self.variables[node.id] = value
    
    def visit_IfStatement(self, node):
        condition = self.visit(node.condition)
        if condition:
            for stmt in node.consequent:
                self.visit(stmt)
        elif node.alternate:
            for stmt in node.alternate:
                self.visit(stmt)
    
    def visit_WhileStatement(self, node):
        loop_guard = 0
        # safety: prevent infinite loops in UI by limiting iterations (large number)
        while self.visit(node.condition):
            for stmt in node.body:
                self.visit(stmt)
            loop_guard += 1
            if loop_guard > 100000:
                raise RuntimeError("Possible infinite loop detected")
    
    def visit_PrintStatement(self, node):
        value = self.visit(node.value)
        self.output.append(value)
    
    def visit_BinaryOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        
        ops = {
            '+': lambda x, y: x + y,
            '-': lambda x, y: x - y,
            '*': lambda x, y: x * y,
            '/': lambda x, y: x / y,
            '<': lambda x, y: 1 if x < y else 0,
            '>': lambda x, y: 1 if x > y else 0,
            '<=': lambda x, y: 1 if x <= y else 0,
            '>=': lambda x, y: 1 if x >= y else 0,
            '==': lambda x, y: 1 if x == y else 0,
            '!=': lambda x, y: 1 if x != y else 0,
        }
        
        if node.op not in ops:
            raise RuntimeError(f"Operator {node.op} not supported")
        
        return ops[node.op](left, right)
    
    def visit_Number(self, node):
        return node.value
    
    def visit_Identifier(self, node):
        if node.id not in self.variables:
            raise NameError(f"Variable '{node.id}' not defined")
        return self.variables[node.id]

# ==================== UTIL: AST -> dict (for UI display) ====================

def ast_to_dict(node):
    if node is None:
        return None
    t = getattr(node, 'type', None)
    if t == 'Program':
        return {'type': 'Program', 'body': [ast_to_dict(s) for s in node.body]}
    if t == 'Declaration':
        return {'type': 'Declaration', 'id': node.id, 'value': ast_to_dict(node.value)}
    if t == 'Assignment':
        return {'type': 'Assignment', 'id': node.id, 'value': ast_to_dict(node.value)}
    if t == 'IfStatement':
        return {'type': 'IfStatement', 'condition': ast_to_dict(node.condition),
                'consequent': [ast_to_dict(s) for s in node.consequent],
                'alternate': [ast_to_dict(s) for s in node.alternate] if node.alternate else None}
    if t == 'WhileStatement':
        return {'type': 'WhileStatement', 'condition': ast_to_dict(node.condition),
                'body': [ast_to_dict(s) for s in node.body]}
    if t == 'PrintStatement':
        return {'type': 'PrintStatement', 'value': ast_to_dict(node.value)}
    if t == 'BinaryOp':
        return {'type': 'BinaryOp', 'op': node.op,
                'left': ast_to_dict(node.left), 'right': ast_to_dict(node.right)}
    if t == 'Number':
        return {'type': 'Number', 'value': node.value}
    if t == 'Identifier':
        return {'type': 'Identifier', 'id': node.id}
    # fallback: try attributes
    return str(node)

# ==================== MAIN COMPILER ====================

class MELCompiler:
    def compile_and_run(self, source_code: str, verbose=False):
        results = {}
        try:
            # Phase 1: Lexical Analysis
            lexer = Lexer(source_code)
            tokens = lexer.tokenize()
            results['tokens'] = tokens

            # Phase 2: Syntax Analysis
            parser = Parser(tokens)
            ast = parser.parse()
            results['ast'] = ast

            # Phase 3: Semantic Analysis
            analyzer = SemanticAnalyzer()
            symbol_table, errors = analyzer.analyze(ast)
            results['symbol_table'] = symbol_table
            results['semantic_errors'] = errors

            if errors:
                # stop further phases if semantic errors exist
                results['three_address_code'] = []
                results['output'] = ''
                return results

            # Phase 4: Intermediate Code Generation
            icg = IntermediateCodeGenerator()
            three_address_code = icg.generate(ast)
            results['three_address_code'] = three_address_code

            # Phase 5 & 6: Execution (Interpreter)
            interpreter = Interpreter()
            output = interpreter.interpret(ast)
            results['output'] = output

            return results

        except Exception as e:
            results['error'] = str(e)
            return results

# ==================== GUI (tkinter) ====================

EXAMPLES = {
    'Factorial': '''# Calculate factorial
let n = 5;
let result = 1;
let i = 1;

while i <= n {
    result = result * i;
    i = i + 1;
}

print result;''',

    'Fibonacci': '''# Fibonacci sequence
let n = 10;
let a = 0;
let b = 1;
let i = 0;

while i < n {
    print a;
    let temp = a + b;
    a = b;
    b = temp;
    i = i + 1;
}''',

    'Conditional': '''let x = 15;

if x > 10 {
    print 1;
} else {
    print 0;
}''',
}

class MELGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("MEL Compiler - GUI")
        self.compiler = MELCompiler()
        self.build_ui()

    def build_ui(self):
        # main paned window
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=1)

        # left frame: code editor and controls
        left = ttk.Frame(paned, padding=6)
        paned.add(left, weight=1)

        lbl = ttk.Label(left, text="MEL Source Code")
        lbl.pack(anchor='w')

        self.code_text = tk.Text(left, wrap='none', width=60, height=30)
        self.code_text.pack(fill=tk.BOTH, expand=1)

        # small control frame
        controls = ttk.Frame(left)
        controls.pack(fill=tk.X, pady=4)

        run_btn = ttk.Button(controls, text="Run", command=self.run_code)
        run_btn.pack(side=tk.LEFT, padx=3)

        load_menu = ttk.Menubutton(controls, text="Load Example")
        menu = tk.Menu(load_menu, tearoff=0)
        for name in EXAMPLES:
            menu.add_command(label=name, command=lambda n=name: self.load_example(n))
        load_menu['menu'] = menu
        load_menu.pack(side=tk.LEFT, padx=3)

        open_btn = ttk.Button(controls, text="Open File", command=self.open_file)
        open_btn.pack(side=tk.LEFT, padx=3)

        save_btn = ttk.Button(controls, text="Save File", command=self.save_file)
        save_btn.pack(side=tk.LEFT, padx=3)

        clear_btn = ttk.Button(controls, text="Clear", command=lambda: self.code_text.delete('1.0', tk.END))
        clear_btn.pack(side=tk.LEFT, padx=3)

        # right frame: results in notebook tabs
        right = ttk.Frame(paned, padding=6)
        paned.add(right, weight=1)

        self.nb = ttk.Notebook(right)
        self.nb.pack(fill=tk.BOTH, expand=1)

        self.tab_tokens = self._make_tab("Tokens")
        self.tab_ast = self._make_tab("AST")
        self.tab_symbol = self._make_tab("Symbol Table / Semantic Errors")
        self.tab_3ac = self._make_tab("3-Address Code")
        self.tab_output = self._make_tab("Output")

        # Insert default example
        self.load_example('Factorial')

    def _make_tab(self, title):
        frame = ttk.Frame(self.nb)
        text = tk.Text(frame, wrap='none')
        text.pack(fill=tk.BOTH, expand=1)
        self.nb.add(frame, text=title)
        return text

    def load_example(self, name):
        self.code_text.delete('1.0', tk.END)
        self.code_text.insert('1.0', EXAMPLES[name])

    def open_file(self):
        fname = filedialog.askopenfilename(title="Open MEL file", filetypes=[("MEL files", "*.mel *.txt"), ("All files", "*.*")])
        if fname:
            try:
                with open(fname, 'r') as f:
                    code = f.read()
                self.code_text.delete('1.0', tk.END)
                self.code_text.insert('1.0', code)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to open file: {e}")

    def save_file(self):
        fname = filedialog.asksaveasfilename(title="Save MEL file", defaultextension=".mel", filetypes=[("MEL files", "*.mel *.txt"), ("All files", "*.*")])
        if fname:
            try:
                with open(fname, 'w') as f:
                    f.write(self.code_text.get('1.0', tk.END))
                messagebox.showinfo("Saved", f"Saved to {fname}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {e}")

    def run_code(self):
        source = self.code_text.get('1.0', tk.END)
        # clear result tabs
        for tab in [self.tab_tokens, self.tab_ast, self.tab_symbol, self.tab_3ac, self.tab_output]:
            tab.delete('1.0', tk.END)

        results = self.compiler.compile_and_run(source, verbose=False)

        # Tokens
        tokens = results.get('tokens', [])
        self.tab_tokens.insert('1.0', '\n'.join(repr(t) for t in tokens))

        # AST
        ast = results.get('ast', None)
        if ast:
            astd = ast_to_dict(ast)
            self.tab_ast.insert('1.0', json.dumps(astd, indent=2))
        else:
            self.tab_ast.insert('1.0', "(no AST)")

        # Symbol table and semantic errors
        st = results.get('symbol_table', {})
        errs = results.get('semantic_errors', [])
        st_text = "Symbol Table:\n" + json.dumps(st, indent=2) + "\n\n"
        st_text += "Semantic Errors:\n" + ("\n".join(errs) if errs else "(none)")
        self.tab_symbol.insert('1.0', st_text)

        # 3AC
        tac = results.get('three_address_code', [])
        if tac:
            self.tab_3ac.insert('1.0', '\n'.join(tac))
        else:
            self.tab_3ac.insert('1.0', "(none)")

        # Output or errors
        if 'error' in results:
            self.tab_output.insert('1.0', "ERROR:\n" + results['error'])
        else:
            output = results.get('output', '')
            self.tab_output.insert('1.0', output if output else "(no output)")

# ==================== RUN APP ====================

def main():
    root = tk.Tk()
    root.geometry("1100x700")
    app = MELGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()



