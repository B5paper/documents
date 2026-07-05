# compiler note

## cache

* 简述 llvm 的用法

    使用 LLVM 实现自定义语言需要多个步骤，下面是一个完整的指南：

    ## 1. **项目架构设计**

    ```
    your-language/
    ├── src/
    │   ├── lexer/      # 词法分析
    │   ├── parser/     # 语法分析
    │   ├── ast/        # 抽象语法树
    │   ├── sema/       # 语义分析
    │   ├── codegen/    # LLVM 代码生成
    │   └── main.cpp    # 主程序
    ├── include/
    ├── tests/
    └── CMakeLists.txt
    ```

    ## 2. **基本实现步骤**

    ### **第 1 步：创建词法分析器（Lexer）**
    ```cpp
    // lexer.h
    #pragma once
    #include <string>
    #include <vector>

    enum TokenType {
        TOK_EOF, TOK_IDENTIFIER, TOK_NUMBER,
        TOK_PLUS, TOK_MINUS, TOK_MULTIPLY, TOK_DIVIDE,
        TOK_LPAREN, TOK_RPAREN, TOK_ASSIGN,
        TOK_IF, TOK_ELSE, TOK_WHILE, TOK_RETURN,
        TOK_INT, TOK_FLOAT
    };

    struct Token {
        TokenType type;
        std::string value;
        int line, column;
    };

    class Lexer {
    public:
        Lexer(const std::string& source);
        Token getNextToken();
    private:
        std::string source;
        size_t position;
        int line, column;
        char currentChar();
        void advance();
        void skipWhitespace();
    };
    ```

    ### **第 2 步：创建语法分析器（Parser）和 AST**
    ```cpp
    // ast.h
    #pragma once
    #include <memory>
    #include <vector>
    #include <string>

    class ASTNode {
    public:
        virtual ~ASTNode() = default;
    };

    class ExprAST : public ASTNode {};
    class StmtAST : public ASTNode {};

    // 表达å节点
    class NumberExprAST : public ExprAST {
    public:
        double value;
        NumberExprAST(double val) : value(val) {}
    };

    class VariableExprAST : public ExprAST {
    public:
        std::string name;
        VariableExprAST(const std::string& name) : name(name) {}
    };

    class BinaryExprAST : public ExprAST {
    public:
        char op;
        std::unique_ptr<ExprAST> LHS, RHS;
        BinaryExprAST(char op, std::unique_ptr<ExprAST> lhs, 
                      std::unique_ptr<ExprAST> rhs)
            : op(op), LHS(std::move(lhs)), RHS(std::move(rhs)) {}
    };

    // 函数节点
    class FunctionAST {
    public:
        std::unique_ptr<PrototypeAST> proto;
        std::unique_ptr<ExprAST> body;
        
        FunctionAST(std::unique_ptr<PrototypeAST> proto,
                    std::unique_ptr<ExprAST> body)
            : proto(std::move(proto)), body(std::move(body)) {}
    };
    ```

    ### **第 3 步：集成 LLVM - 代码生成器**
    ```cpp
    // codegen.h
    #pragma once
    #include "llvm/IR/IRBuilder.h"
    #include "llvm/IR/LLVMContext.h"
    #include "llvm/IR/Module.h"
    #include "llvm/IR/Verifier.h"
    #include <map>

    class CodeGen {
    public:
        CodeGen();
        llvm::Value* codegen(ExprAST* ast);
        llvm::Function* codegen(FunctionAST* ast);
        
    private:
        std::unique_ptr<llvm::LLVMContext> context;
        std::unique_ptr<llvm::IRBuilder<>> builder;
        std::unique_ptr<llvm::Module> module;
        std::map<std::string, llvm::Value*> namedValues;
        
        llvm::Value* logErrorV(const char* str);
        llvm::AllocaInst* createEntryBlockAlloca(llvm::Function* function,
                                                 const std::string& varName);
    };
    ```

    ### **第 4 步：实现代码生成逻辑**
    ```cpp
    // codegen.cpp
    #include "codegen.h"
    #include "ast.h"

    CodeGen::CodeGen() {
        context = std::make_unique<llvm::LLVMContext>();
        module = std::make_unique<llvm::Module>("MyLanguage", *context);
        builder = std::make_unique<llvm::IRBuilder<>>(*context);
    }

    llvm::Value* CodeGen::codegen(NumberExprAST* num) {
        return llvm::ConstantFP::get(*context, llvm::APFloat(num->value));
    }

    llvm::Value* CodeGen::codegen(VariableExprAST* var) {
        llvm::Value* V = namedValues[var->name];
        if (!V) {
            // 错误处理：未定义的变量
            return logErrorV("Unknown variable name");
        }
        return V;
    }

    llvm::Value* CodeGen::codegen(BinaryExprAST* bin) {
        llvm::Value* L = codegen(bin->LHS.get());
        llvm::Value* R = codegen(bin->RHS.get());
        if (!L || !R) return nullptr;
        
        switch (bin->op) {
            case '+': return builder->CreateFAdd(L, R, "addtmp");
            case '-': return builder->CreateFSub(L, R, "subtmp");
            case '*': return builder->CreateFMul(L, R, "multmp");
            case '/': return builder->CreateFDiv(L, R, "divtmp");
            default:
                return logErrorV("Invalid binary operator");
        }
    }

    llvm::Function* CodeGen::codegen(FunctionAST* func) {
        // 创建函数类型：double func()
        llvm::FunctionType* FT = llvm::FunctionType::get(
            llvm::Type::getDoubleTy(*context), false);
        
        llvm::Function* F = llvm::Function::Create(
            FT, llvm::Function::ExternalLinkage,
            func->proto->name, module.get());
        
        // 创建基本块
        llvm::BasicBlock* BB = llvm::BasicBlock::Create(*context, "entry", F);
        builder->SetInsertPoint(BB);
        
        // 生成函数体
        namedValues.clear();
        llvm::Value* retVal = codegen(func->body.get());
        if (!retVal) {
            F->eraseFromParent();
            return nullptr;
        }
        
        // 创建返回指令
        builder->CreateRet(retVal);
        
        // 验证函数
        llvm::verifyFunction(*F);
        
        return F;
    }
    ```

    ### **第 5 步：主程序入口**
    ```cpp
    // main.cpp
    #include "lexer.h"
    #include "parser.h"
    #include "codegen.h"
    #include <fstream>
    #include <iostream>

    int main(int argc, char* argv[]) {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <source_file>" << std::endl;
            return 1;
        }
        
        // 读取源文件
        std::ifstream file(argv[1]);
        std::string source((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        
        // 词法分析
        Lexer lexer(source);
        
        // 语法分析
        Parser parser(lexer);
        auto ast = parser.parse();
        
        if (!ast) {
            std::cerr << "Parse failed!" << std::endl;
            return 1;
        }
        
        // 代码生成
        CodeGen generator;
        generator.codegen(ast.get());
        
        // 输出 LLVM IR
        generator.dumpIR();
        
        return 0;
    }
    ```

    ## 3. **CMake 配置**
    ```cmake
    cmake_minimum_required(VERSION 3.10)
    project(MyLanguage)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # 查找 LLVM
    find_package(LLVM REQUIRED CONFIG)
    message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
    message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

    include_directories(${LLVM_INCLUDE_DIRS})
    add_definitions(${LLVM_DEFINITIONS})

    # 添加 LLVM 组件
    llvm_map_components_to_libnames(llvm_libs
        core
        irreader
        mcjit
        native
        support
    )

    add_executable(mylang
        src/main.cpp
        src/lexer/lexer.cpp
        src/parser/parser.cpp
        src/codegen/codegen.cpp
    )

    target_link_libraries(mylang ${llvm_libs})
    ```

    ## 4. **编译和运行示例**

    ### 创建示例语言程序
    ```c
    // example.mylang
    def main() {
        x = 10 + 5 * 2;
        y = (x - 5) / 3;
        return y;
    }
    ```

    ### 编译你的编译器
    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```

    ### 运行编译器
    ```bash
    # 生成 LLVM IR
    ./mylang example.mylang > output.ll

    # 使用 clang 编译成可执行文件
    clang output.ll -o example

    # 运行程序
    ./example
    echo $?  # 查看返回值
    ```

    ## 5. **进阶功能**

    ### **添加 JIT 支持**
    ```cpp
    #include "llvm/ExecutionEngine/ExecutionEngine.h"
    #include "llvm/ExecutionEngine/GenericValue.h"

    class JITCompiler {
    public:
        llvm::GenericValue runFunction(llvm::Function* F,
                                       const std::vector<llvm::GenericValue>& args) {
            std::string errStr;
            llvm::ExecutionEngine* EE = llvm::EngineBuilder(std::move(module))
                .setErrorStr(&errStr)
                .create();
            
            if (!EE) {
                llvm::errs() << "Failed to create ExecutionEngine: " << errStr << "\n";
                return llvm::GenericValue();
            }
            
            llvm::GenericValue result = EE->runFunction(F, args);
            delete EE;
            return result;
        }
    };
    ```

    ### **添加控制流（if/while）**
    ```cpp
    llvm::Value* CodeGen::codegen(IfExprAST* ifExpr) {
        llvm::Value* cond = codegen(ifExpr->cond.get());
        cond = builder->CreateFCmpONE(
            cond, llvm::ConstantFP::get(*context, llvm::APFloat(0.0)), "ifcond");
        
        llvm::Function* F = builder->GetInsertBlock()->getParent();
        llvm::BasicBlock* thenBB = llvm::BasicBlock::Create(*context, "then", F);
        llvm::BasicBlock* elseBB = llvm::BasicBlock::Create(*context, "else");
        llvm::BasicBlock* mergeBB = llvm::BasicBlock::Create(*context, "ifcont");
        
        builder->CreateCondBr(cond, thenBB, elseBB);
        
        // 生成 then 块
        builder->SetInsertPoint(thenBB);
        llvm::Value* thenVal = codegen(ifExpr->thenExpr.get());
        builder->CreateBr(mergeBB);
        thenBB = builder->GetInsertBlock();
        
        // 生成 else 块
        F->getBasicBlockList().push_back(elseBB);
        builder->SetInsertPoint(elseBB);
        llvm::Value* elseVal = codegen(ifExpr->elseExpr.get());
        builder->CreateBr(mergeBB);
        elseBB = builder->GetInsertBlock();
        
        // 合并块
        F->getBasicBlockList().push_back(mergeBB);
        builder->SetInsertPoint(mergeBB);
        
        // 创建 phi 节点
        llvm::PHINode* PN = builder->CreatePHI(
            llvm::Type::getDoubleTy(*context), 2, "iftmp");
        PN->addIncoming(thenVal, thenBB);
        PN->addIncoming(elseVal, elseBB);
        
        return PN;
    }
    ```

    ## 6. **学习资源**

    1. **官方教程**：Kaleidoscope 教程（LLVM 官方）
    2. **书籍**：《Getting Started with LLVM Core Libraries》
    3. **参考实现**：
       - [Kaleidoscope 实现](https://llvm.org/docs/tutorial/)
       - [TinyPython](https://github.com/rui314/minilisp)
       - [Simple Language](https://github.com/eliben/llvm-clang-samples)

    ## 7. **调试工具**
    ```bash
    # 查看生成的 IR
    opt -S -O2 output.ll -o optimized.ll

    # 可视化控制流
    opt -dot-cfg output.ll
    dot -Tpng .main.dot -o cfg.png

    # 使用 LLVM 调试器
    lli -jit-kind=mcjit output.ll
    ```

    实现自定义语言的关键是逐步构建：从词法分析 → 语法分析 → AST → 语义分析 → LLVM IR 生成。建议先实现简单的算术表达式，再逐步添加变量、函数、控制流等特性。


* 那如果我仅想用 llvm 分析 js 或 c/c++ 或 markdown 的 ast，怎么做

    使用 LLVM 分析现有语言的 AST 有几种不同的方法，下面分别针对不同语言说明：

    ## 1. **分析 C/C++ AST - 使用 Clang**

    ### **方法 1：使用 libclang（推荐）**
    ```cpp
    // analyze_c_ast.cpp
    #include <clang-c/Index.h>
    #include <iostream>
    #include <vector>

    // AST 访问者回调函数
    CXChildVisitResult visitor(CXCursor cursor, CXCursor parent, CXClientData client_data) {
        CXSourceLocation loc = clang_getCursorLocation(cursor);
        if (clang_Location_isInSystemHeader(loc)) {
            return CXChildVisit_Continue;
        }
        
        // 获取光标信息
        CXString cursorKind = clang_getCursorKindSpelling(clang_getCursorKind(cursor));
        CXString cursorName = clang_getCursorSpelling(cursor);
        CXString cursorType = clang_getTypeSpelling(clang_getCursorType(cursor));
        
        // 打印 AST 节点信息
        std::cout << "Cursor: " << clang_getCString(cursorKind)
                  << " | Name: " << clang_getCString(cursorName)
                  << " | Type: " << clang_getCString(cursorType) << std::endl;
        
        clang_disposeString(cursorKind);
        clang_disposeString(cursorName);
        clang_disposeString(cursorType);
        
        // 递归访问子节点
        clang_visitChildren(cursor, visitor, client_data);
        
        return CXChildVisit_Continue;
    }

    int main(int argc, char** argv) {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <source_file.cpp>" << std::endl;
            return 1;
        }
        
        // 创建索引
        CXIndex index = clang_createIndex(0, 0);
        
        // 解析文件
        const char* args[] = {"-std=c++17"};
        CXTranslationUnit unit = clang_parseTranslationUnit(
            index,
            argv[1],
            args, 1,
            nullptr, 0,
            CXTranslationUnit_None
        );
        
        if (!unit) {
            std::cerr << "Failed to parse translation unit" << std::endl;
            return 1;
        }
        
        // 获取顶层 AST 节点
        CXCursor cursor = clang_getTranslationUnitCursor(unit);
        
        // 遍历 AST
        std::cout << "=== AST Dump ===" << std::endl;
        clang_visitChildren(cursor, visitor, nullptr);
        
        // 获取诊断信息
        unsigned diagnosticCount = clang_getNumDiagnostics(unit);
        std::cout << "\n=== Diagnostics (" << diagnosticCount << ") ===" << std::endl;
        for (unsigned i = 0; i < diagnosticCount; ++i) {
            CXDiagnostic diag = clang_getDiagnostic(unit, i);
            CXString diagStr = clang_formatDiagnostic(diag, clang_defaultDiagnosticDisplayOptions());
            std::cout << clang_getCString(diagStr) << std::endl;
            clang_disposeString(diagStr);
            clang_disposeDiagnostic(diag);
        }
        
        // 清理
        clang_disposeTranslationUnit(unit);
        clang_disposeIndex(index);
        
        return 0;
    }
    ```

    ### **方法 2：使用 Clang AST Matchers（更强大的分析）**
    ```cpp
    // ast_matcher.cpp
    #include <clang/ASTMatchers/ASTMatchers.h>
    #include <clang/ASTMatchers/ASTMatchFinder.h>
    #include <clang/Tooling/Tooling.h>
    #include <clang/Frontend/CompilerInstance.h>

    using namespace clang;
    using namespace clang::ast_matchers;
    using namespace clang::tooling;

    class FunctionPrinter : public MatchFinder::MatchCallback {
    public:
        virtual void run(const MatchFinder::MatchResult &Result) {
            if (const FunctionDecl* func = Result.Nodes.getNodeAs<FunctionDecl>("function")) {
                // 获取函数信息
                SourceManager& sm = *Result.SourceManager;
                SourceLocation loc = func->getLocation();
                
                std::cout << "Found function: " << func->getNameAsString() 
                          << " at " << sm.getFilename(loc).str()
                          << ":" << sm.getSpellingLineNumber(loc) << std::endl;
                
                // 打印参数
                for (auto param : func->parameters()) {
                    std::cout << "  Param: " << param->getNameAsString()
                             << " : " << param->getType().getAsString() << std::endl;
                }
            }
            
            if (const VarDecl* var = Result.Nodes.getNodeAs<VarDecl>("globalVar")) {
                std::cout << "Global variable: " << var->getNameAsString()
                          << " : " << var->getType().getAsString() << std::endl;
            }
        }
    };

    int main(int argc, char** argv) {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <source_file.cpp>" << std::endl;
            return 1;
        }
        
        std::string code = R"cpp(
            #include <iostream>
            
            int global_var = 42;
            
            void hello_world() {
                std::cout << "Hello, World!" << std::endl;
            }
            
            int add(int a, int b) {
                return a + b;
            }
        )cpp";
        
        // 创建匹配器
        FunctionPrinter printer;
        MatchFinder finder;
        
        // 定义匹配规则
        finder.addMatcher(
            functionDecl(unless(isExpansionInSystemHeader()))
                .bind("function"),
            &printer);
        
        finder.addMatcher(
            varDecl(hasGlobalStorage(), unless(isExpansionInSystemHeader()))
                .bind("globalVar"),
            &printer);
        
        // 创建并运行工具
        std::vector<std::string> args = {"-std=c++17"};
        ClangTool tool(getCompilationDatabaseFromCommandLine(args), {argv[1]});
        
        return tool.run(newFrontendActionFactory(&finder).get());
    }
    ```

    ### **编译 Clang 工具**
    ```bash
    # CMakeLists.txt
    cmake_minimum_required(VERSION 3.10)
    project(ASTAnalyzer)

    find_package(Clang REQUIRED)
    include_directories(${CLANG_INCLUDE_DIRS})

    add_executable(analyze_c_ast analyze_c_ast.cpp)
    target_link_libraries(analyze_c_ast clang-cpp)

    add_executable(ast_matcher ast_matcher.cpp)
    target_link_libraries(ast_matcher
        clangASTMatchers
        clangTooling
        clangBasic
        clangAST
    )
    ```

    ## 2. **分析 JavaScript AST**

    ### **方法 1：使用 JavaScript 解析库生成 AST，然后用 LLVM 分析**
    ```cpp
    // 先使用 QuickJS 或 V8 解析 JavaScript
    // 然后转换为 LLVM 可以处理的格式

    #include <iostream>
    #include <fstream>
    #include <string>

    // 使用 QuickJS 解析 JavaScript
    extern "C" {
    #include "quickjs.h"
    #include "quickjs-libc.h"
    }

    void analyze_js_ast(const std::string& js_code) {
        JSRuntime* rt = JS_NewRuntime();
        JSContext* ctx = JS_NewContext(rt);
        
        // 解析 JavaScript
        const char* filename = "input.js";
        JSValue result = JS_Eval(ctx, js_code.c_str(), js_code.length(),
                                filename, JS_EVAL_FLAG_COMPILE_ONLY);
        
        if (JS_IsException(result)) {
            JSValue exception = JS_GetException(ctx);
            const char* err_str = JS_ToCString(ctx, exception);
            std::cerr << "Parse error: " << err_str << std::endl;
            JS_FreeCString(ctx, err_str);
        } else {
            // 获取字节码并分析
            size_t bc_len;
            uint8_t* bc_buf = JS_WriteObject(ctx, &bc_len, result, JS_WRITE_OBJ_BYTECODE);
            
            if (bc_buf) {
                // 这里可以分析字节码结构
                std::cout << "Generated bytecode of length: " << bc_len << std::endl;
                js_free(ctx, bc_buf);
            }
        }
        
        JS_FreeValue(ctx, result);
        JS_FreeContext(ctx);
        JS_FreeRuntime(rt);
    }
    ```

    ### **方法 2：使用 Babel/Esprima 生成 JSON AST，然后用 C++ 分析**
    ```python
    # 先用 Python 预处理，生成标准 AST
    import json
    import subprocess
    import sys

    def get_js_ast(js_file):
        # 使用 esprima 解析 JavaScript
        esprima_code = f"""
        const esprima = require('esprima');
        const fs = require('fs');
        const code = fs.readFileSync('{js_file}', 'utf-8');
        const ast = esprima.parseScript(code, {{ 
            tokens: true, 
            comment: true,
            loc: true,
            range: true,
            tolerant: true
        }});
        console.log(JSON.stringify(ast, null, 2));
        """
        
        result = subprocess.run(['node', '-e', esprima_code], 
                              capture_output=True, text=True)
        return json.loads(result.stdout)

    # 然后在 C++ 中分析这个 JSON AST
    ```

    ```cpp
    // analyze_js_json.cpp
    #include <iostream>
    #include <fstream>
    #include <nlohmann/json.hpp>

    using json = nlohmann::json;

    void traverse_js_ast(const json& node, int depth = 0) {
        if (!node.is_object()) return;
        
        std::string indent(depth * 2, ' ');
        
        // 打印节点信息
        if (node.contains("type")) {
            std::cout << indent << "Type: " << node["type"] << std::endl;
            
            if (node["type"] == "VariableDeclaration") {
                std::cout << indent << "  Kind: " << node["kind"] << std::endl;
            }
            
            if (node.contains("loc")) {
                std::cout << indent << "  Location: " 
                         << node["loc"]["start"]["line"] << ":" 
                         << node["loc"]["start"]["column"] << std::endl;
            }
        }
        
        // 递归遍历子节点
        for (auto& [key, value] : node.items()) {
            if (key == "type" || key == "loc" || key == "range") continue;
            
            if (value.is_array()) {
                std::cout << indent << "  " << key << ": [" << std::endl;
                for (auto& child : value) {
                    traverse_js_ast(child, depth + 2);
                }
                std::cout << indent << "  ]" << std::endl;
            } else if (value.is_object()) {
                std::cout << indent << "  " << key << ": {" << std::endl;
                traverse_js_ast(value, depth + 2);
                std::cout << indent << "  }" << std::endl;
            }
        }
    }

    int main() {
        std::ifstream file("ast.json");
        json ast = json::parse(file);
        traverse_js_ast(ast);
        return 0;
    }
    ```

    ## 3. **分析 Markdown AST**

    ### **使用 CommonMark 解析器**
    ```cpp
    // analyze_markdown.cpp
    #include <iostream>
    #include <fstream>
    #include <string>
    #include <vector>
    #include <cmark.h>

    // 定义 AST 节点结构
    struct MarkdownNode {
        enum Type {
            DOCUMENT, PARAGRAPH, TEXT, HEADING,
            LIST, LIST_ITEM, CODE_BLOCK, LINK
        };
        
        Type type;
        std::string content;
        int level;  // 用于 heading 级别
        std::string url;  // 用于 link
        std::vector<MarkdownNode*> children;
        
        ~MarkdownNode() {
            for (auto child : children) delete child;
        }
    };

    class MarkdownAnalyzer {
    public:
        MarkdownAnalyzer(const std::string& filename) {
            std::ifstream file(filename);
            std::string content((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
            ast_root = parseMarkdown(content);
        }
        
        void analyze() {
            std::cout << "=== Markdown Analysis ===" << std::endl;
            std::cout << "Total headings: " << countHeadings() << std::endl;
            std::cout << "Total links: " << countLinks() << std::endl;
            std::cout << "Total code blocks: " << countCodeBlocks() << std::endl;
            
            // 打印大纲
            std::cout << "\n=== Document Outline ===" << std::endl;
            printOutline(ast_root, 0);
        }
        
    private:
        MarkdownNode* ast_root;
        
        MarkdownNode* parseMarkdown(const std::string& content) {
            // 使用 CommonMark 解析
            cmark_parser* parser = cmark_parser_new(CMARK_OPT_DEFAULT);
            cmark_parser_feed(parser, content.c_str(), content.size());
            cmark_node* doc = cmark_parser_finish(parser);
            cmark_parser_free(parser);
            
            // 转换为自定义 AST
            MarkdownNode* root = convertCmarkToAST(doc);
            cmark_node_free(doc);
            
            return root;
        }
        
        MarkdownNode* convertCmarkToAST(cmark_node* node) {
            if (!node) return nullptr;
            
            MarkdownNode* md_node = new MarkdownNode();
            
            // 转换节点类型
            switch (cmark_node_get_type(node)) {
                case CMARK_NODE_DOCUMENT:
                    md_node->type = MarkdownNode::DOCUMENT;
                    break;
                case CMARK_NODE_PARAGRAPH:
                    md_node->type = MarkdownNode::PARAGRAPH;
                    break;
                case CMARK_NODE_TEXT:
                    md_node->type = MarkdownNode::TEXT;
                    md_node->content = cmark_node_get_literal(node);
                    break;
                case CMARK_NODE_HEADING:
                    md_node->type = MarkdownNode::HEADING;
                    md_node->level = cmark_node_get_heading_level(node);
                    break;
                case CMARK_NODE_LINK:
                    md_node->type = MarkdownNode::LINK;
                    md_node->url = cmark_node_get_url(node);
                    break;
                case CMARK_NODE_CODE_BLOCK:
                    md_node->type = MarkdownNode::CODE_BLOCK;
                    md_node->content = cmark_node_get_literal(node);
                    break;
                default:
                    // 处理其他节点类型
                    break;
            }
            
            // 递归处理子节点
            cmark_node* child = cmark_node_first_child(node);
            while (child) {
                MarkdownNode* child_ast = convertCmarkToAST(child);
                if (child_ast) {
                    md_node->children.push_back(child_ast);
                }
                child = cmark_node_next(child);
            }
            
            return md_node;
        }
        
        int countHeadings() const { return countNodes(ast_root, MarkdownNode::HEADING); }
        int countLinks() const { return countNodes(ast_root, MarkdownNode::LINK); }
        int countCodeBlocks() const { return countNodes(ast_root, MarkdownNode::CODE_BLOCK); }
        
        int countNodes(MarkdownNode* node, MarkdownNode::Type type) const {
            if (!node) return 0;
            
            int count = (node->type == type) ? 1 : 0;
            for (auto child : node->children) {
                count += countNodes(child, type);
            }
            return count;
        }
        
        void printOutline(MarkdownNode* node, int depth) {
            if (!node) return;
            
            std::string indent(depth * 2, ' ');
            
            if (node->type == MarkdownNode::HEADING) {
                std::cout << indent << "H" << node->level << ": ";
                // 提取标题文本
                for (auto child : node->children) {
                    if (child->type == MarkdownNode::TEXT) {
                        std::cout << child->content;
                    }
                }
                std::cout << std::endl;
            }
            
            for (auto child : node->children) {
                printOutline(child, depth + 1);
            }
        }
    };

    int main(int argc, char** argv) {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <markdown_file>" << std::endl;
            return 1;
        }
        
        MarkdownAnalyzer analyzer(argv[1]);
        analyzer.analyze();
        
        return 0;
    }
    ```

    ### **编译 Markdown 分析器**
    ```bash
    # 安装 CommonMark
    git clone https://github.com/commonmark/cmark.git
    cd cmark
    make
    sudo make install

    # 编译
    g++ -std=c++11 analyze_markdown.cpp -o analyze_markdown -lcmark
    ```

    ## 4. **通用 AST 分析框架**

    ### **设计通用 AST 接口**
    ```cpp
    // generic_ast.h
    #pragma once
    #include <string>
    #include <vector>
    #include <memory>
    #include <map>

    class ASTNode {
    public:
        enum NodeType {
            NT_UNKNOWN,
            NT_FUNCTION, NT_VARIABLE, NT_CLASS,
            NT_CALL, NT_LOOP, NT_CONDITIONAL,
            NT_LITERAL, NT_IDENTIFIER
        };
        
        NodeType type;
        std::string name;
        std::string value;
        std::vector<std::unique_ptr<ASTNode>> children;
        std::map<std::string, std::string> attributes;
        
        ASTNode(NodeType t = NT_UNKNOWN) : type(t) {}
        virtual ~ASTNode() = default;
        
        void addChild(std::unique_ptr<ASTNode> child) {
            children.push_back(std::move(child));
        }
    };

    class ASTAnalyzer {
    public:
        virtual std::unique_ptr<ASTNode> parse(const std::string& source) = 0;
        virtual void analyze(const ASTNode* root) = 0;
        virtual void transform(ASTNode* root) = 0;
        virtual void generate(const ASTNode* root, const std::string& output_file) = 0;
    };

    // 具体的语言分析器
    class CppASTAnalyzer : public ASTAnalyzer {
    public:
        std::unique_ptr<ASTNode> parse(const std::string& source) override;
        void analyze(const ASTNode* root) override;
        void transform(ASTNode* root) override;
        void generate(const ASTNode* root, const std::string& output_file) override;
    };

    class JSASTAnalyzer : public ASTAnalyzer {
    public:
        std::unique_ptr<ASTNode> parse(const std::string& source) override;
        void analyze(const ASTNode* root) override;
        void transform(ASTNode* root) override;
        void generate(const ASTNode* root, const std::string& output_file) override;
    };
    ```

    ## 5. **实用工具和命令**

    ### **直接使用 Clang 工具**
    ```bash
    # 查看 C/C++ AST
    clang -Xclang -ast-dump -fsyntax-only example.cpp

    # 导出为 JSON
    clang -Xclang -ast-dump=json -fsyntax-only example.cpp > ast.json

    # 使用 clang-query 交互式查询
    clang-query example.cpp -- -std=c++17
    # 然后在 clang-query 中输入: match functionDecl()
    ```

    ### **使用 LLVM 进行静态分析**
    ```bash
    # 使用 clang-tidy 进行代码检查
    clang-tidy example.cpp --checks='*' -- -std=c++17

    # 使用 scan-build 进行静态分析
    scan-build clang++ example.cpp -o example
    ```

    ## 6. **集成 LLVM Pass 进行分析**

    ```cpp
    // custom_pass.cpp
    #include "llvm/IR/Function.h"
    #include "llvm/IR/Instructions.h"
    #include "llvm/IR/LegacyPassManager.h"
    #include "llvm/Pass.h"
    #include "llvm/Support/raw_ostream.h"

    using namespace llvm;

    namespace {
        class FunctionInfoPass : public FunctionPass {
        public:
            static char ID;
            FunctionInfoPass() : FunctionPass(ID) {}
            
            bool runOnFunction(Function &F) override {
                errs() << "Function: " << F.getName() << "\n";
                errs() << "  Number of arguments: " << F.arg_size() << "\n";
                errs() << "  Number of basic blocks: " << F.size() << "\n";
                errs() << "  Number of instructions: " << F.getInstructionCount() << "\n";
                
                // 分析指令类型
                int numCalls = 0, numLoads = 0, numStores = 0;
                for (auto& BB : F) {
                    for (auto& I : BB) {
                        if (isa<CallInst>(&I)) numCalls++;
                        if (isa<LoadInst>(&I)) numLoads++;
                        if (isa<StoreInst>(&I)) numStores++;
                    }
                }
                
                errs() << "  Call instructions: " << numCalls << "\n";
                errs() << "  Load instructions: " << numLoads << "\n";
                errs() << "  Store instructions: " << numStores << "\n";
                
                return false;  // 没有修改函数
            }
        };
    }

    char FunctionInfoPass::ID = 0;
    static RegisterPass<FunctionInfoPass> X("func-info", "Function Information Pass");

    // 编译和使用
    // opt -load ./CustomPass.so -func-info < input.bc > /dev/null
    ```

    ## **总结建议**

    1. **对于 C/C++**：直接使用 Clang，它已经提供了完整的 AST
    2. **对于 JavaScript**：
       - 使用 QuickJS/V8 解析，然后转换为自己的 AST 格式
       - 或用 Node.js 生成标准 ESTree AST，再在 C++ 中分析
    3. **对于 Markdown**：使用 CommonMark 等成熟解析器
    4. **通用方法**：设计统一的 AST 接口，为每种语言实现适配器

    **最推荐的方式**：
    - 对于 C/C++：直接用 Clang 的 libclang 或 AST Matchers
    - 对于其他语言：先用该语言的最佳解析器生成 AST，再设计通用接口用 C++ 分析

* 和`%{`配对的是`%}`，不是`}%`。百分号永远在前。

    ```c
    %{
    int num_cnt = 0;
    %}
    ```

* flex 中，`\n {return 0;}`表示结束 parser 程序，进入主程序。如果写成`\n {}`，那么即使按回车换行，parser 程序也不结束。

* compiler: 文件结束符

    文件结束符$\$$被规定出现在 expression $E$之后，通常记为$S \rightarrow E\$$。

    任意字符串都可以作为文件结束符吗，比如`EOF`，还是说只有一个特殊字符`\0`可以作为文件结束符？

    如果将任意字符串作为文件结束符，是否会引入冲突？后面学完整个文法分析，可以对这个问题做个尝试。

* question: 对于图 2-4，如果输入了两个`-`，即`--`，那么并不会在 10 处终止，但是根据最长匹配原则，又不能只匹配一个`-`就停下。该如何解释这个矛盾？

* $\mathbf{edge}(s, c)$ 的解释

	$s$是集合中的一个元素，或者说是状态机有向图中的一个节点。

	$c$是一条边对应的字母，或者条件。

	$\mathbf{edge}(s, c)$指的是从状态$s$沿着标有$c$的边可到达的所有 NFA 节点的集合。

	$\mathbf{edge}(s, c)$是**节点的集合**。

	example:

	<div style='text-align:center'>
	<img src='../Reference_resources/ref_9/pics/2024.03.15/pic_1.jpg'>
	</div>

	在上图中，$\mathbf{edge}(1, c) = \{2, 4\}$

* $\mathbf{closure}(S)$的解释

	对于状态集合$S$，$\mathbf{closure}(S)$是满足如下条件的最小集合$T$：

	$T = S\cup\left( \bigcup\limits_{s \in T} \mathbf{edge}(s, \epsilon) \right)$

	其实这个表示的就是从$S$出发，能通过空字符串条件达到的所有状态。

	注意，这里对$T$的定义使用了递归。我们从$S$出发，经过$\epsilon$边，到达一些状态，再从这些状态出发，再经过$\epsilon$边，到达一些新的状态，这样不断迭代下去，直到算法停止。这个递归挺新颖的。

	使用伪代码描述上面的公式，就是

	$\begin{aligned}
	&T \leftarrow S \\
	&\mathbf{repeat}\ T' \leftarrow T \\
	&\hphantom{asdf} T \leftarrow T' \cup \left(\bigcup_{s \in T'} \mathbf{edge}(s, \epsilon) \right) \\
	&\mathbf{until}\ T = T'
	\end{aligned}$

* lex/flex hello world program

	`count_num.lex`:

	```lex
	%{
	int digit_count = 0;
	%}

	%%
	[0-9]+ {digit_count++;}
	. {}
	\n {return 0;}
	%%

	int yywrap(){}

	int main()
	{
		yylex();
		printf("total %d numbers.\n", digit_count);
		return 0;
	}
	```

	使用 lex/flex 程序处理：

	```bash
	flex count_num.lex
	```

	此时会生成一个`lex.yy.c`文件。

	然后再调用`gcc`编译器：

	```bash
	gcc lex.yy.c -o count_num
	```

	执行：

	```
	./count_num
	```

	此时会进入 stdin 模式等待输入。

	input:

	```
	123 345 hehe #@!
	```

	按 enter 键后，输出为

	```
	total 2 numbers.
	```

	flex tutorial: <https://www.geeksforgeeks.org/flex-fast-lexical-analyzer-generator/>

	flex project site: <https://github.com/westes/flex>

	unread:
	
	1. <https://begriffs.com/posts/2021-11-28-practical-parsing.html>

	2. <https://web.mit.edu/gnu/doc/html/flex_1.html>

* compiler 最长匹配

    在最长匹配时，需要保存三个状态位置，一个是本次匹配的起始位置，一个是上次的终态位置，还有一个是当前位置。

    如果下一个位置找不到有限状态机的出口边，那么就认为此次匹配结束，上次终态的位置为最终匹配结果，并以此更新起始位置。

    如果下一个位置找到了状态机对应的出口边，并且没有终态，那么继续往后找；如果找到了终态，那么更新上次终态的位置。

* 编译原理图 2-3

    <div style='text-align:center'>
    <img src="../Reference_resources/ref_4/pic_0.png">
    </div>

    REAL 那一个，为什么节点 4 不能直接连到节点 3？

    white space 没看懂，难道注释指的是`--hello,world`这样的形式吗？

    error 也没看懂，它的意思是说，如果有一个字符，只要这个字符不是`\n`，就说明错误的意思吗？

    ID 那个图，是否包含了 IF 的情况？

## 词法

### 正则表达式 regular expression

正则表达式定义了一个由字符串组成的集合。正则表达式可以使用有限的符号来定义无限元素的集合。

* 符号（symbol）

    symbol $\mathbf a$ 可以匹配任意包含$\mathbf a$的字符串。

* 可选（alternation）

    对于两个正则表达式$M$，$N$，$M | N$可以形成一个新的正则表达式。只要一个字符串包含$M$或$N$其中的一个，就可以匹配得上。

* 联结（concatenation）

    $M \cdot N$表示$M$后紧跟$N$。

* $\epsilon$（epsilon）

    表示空字符串。

* 重复（repetition）

    对于给定的正则表达式$M$，它的克林（Kleene）闭包是$M^*$。

    如果一个字符串是由$M$中的字符串经零至多次联结运算的结果，则该字符串属于$M^*$。

Examples:

* $a$

    表示`{"a"}`。

* $a\, |\, b$

    表示`{"a", "b"}`。

* $(a\, |\, b) \cdot a$

    表示`{"aa", "ba"}`。

* $(a \cdot b) \, |\, \epsilon$

    表示`{"", "ab"}`。

* $((a\, |\, b) \cdot a)^*$

    表示`{"", "aa", "ba", "aaaa", "baaa", "aaba", "baba", "aaaaaa", ...}`

* $(0 \, | \, 1)^* \cdot 0$

    由$2$的倍数组成的二进制数。

* $b^*(abb^*)^*(a \, | \, \epsilon)$

    由`a`和`b`组成，但`a`不连续出现的字符串。

* $(a\, | \, b)^*aa(a \, | \, b)^*$

    由`a`和`b`组成，且有连续出现的`a`的字符串。

在写正则表达式时，联结符号（$\cdot$）和$\epsilon$符号可以被省略。并假定克林闭包的优先级高于联结运算，联结运算的优先级高于可选运算。

examples:

* $ab \, | \, c$表示$(a \cdot b) \, | \, c$

* $(a\, |\, )$表示$(a \, | \, \epsilon)$

感觉这样定义的正则表达式其实就是离散数学和集合论的结合，每个位置上可能有，可能没有，再加𣥖与，或，空集。

* 正则表达式的一些缩写

    `[abcd]`表示`(a|b|c|d)`

    `[b-g]`表示`[bcdefg]`

    `[b-gM-Qkr]`表示`[bcdefgMNOPQkr]`

    $M?$表示$(M | \epsilon)$

    $M^+$表示$(M \cdot M^*)$

    这些缩写只是为了方便，并没有增加正则表达式的描述能力。

#### 正则表达式的一些 example

| 正则表达式 | 对应的字符串类型 |
| - | - |
| `if` | `{return IF;}` |
| `[a-z][a-z0-9]*` | `{return ID;}` |
| `[0-9]+` | `{return NUM;}` |
| `([0-9]+"."[0-9]*)|([0-9]*"."[0-9]+)` | `{return REAL;}` |
| `("--"[a-z]*"\n")|(" "|"\n"|"\t")+` | `{ /* do nothing */ }`|
| `.` | `{ error(); }` |

这里比较有意思的是`REAL`这一行。正则表达式有点像填空，每个位置上，一个 pattern 可能出现零次，一次，或多次。但是一个位置上 pattern 的出现与否，并不影响另外位置上的 pattern 是否出现。因此如果要约束两个位置上只能是`(0, 1), (1, 0), (1, 1)`，而不能是`(0, 0)`，那么就只能写两遍，就像 real 那一行那样。

注意，最后一行的`.`并不是字符串句号，而是句点符号，表示除換行符之外的任意单个字符。

* $(0 | 1)^* \cdot 0$

    由 2 的倍数组成的二进制数

* $b^*(abb^*)^*(a | \epsilon)$

    由$a$和$b$组成，但$a$不连续出现的字符串。

* $(a | b)^*aa(a | b)^*$

    由$a$和$b$组成，且有连续出现的$a$的字符串。

    解释：这个连续出现$a$表示至少出现 2 次。也就是说，至少有一个$aa$出现，且$aa$出现的前置条件可以是$a$，也可以是$b$。因此前置条件写成$(a | b)^*$，后置条件并不会对题目要求产生影响，因此可以是任意字符串，所以填$(a | b)^*$。


* `{a, b}`中不包含连续字符串`baa`的正则表达式

	$a*(b(abb*)*(a|\epsilon))*(b|\epsilon)$

	一些构造正则表达式的经验：

	1. a 不连续出现的情况的匹配指的是匹配所有a不连续出现的非重叠字符串。

		也就是说，不是只匹配整个字符串，每个不重叠子串也应该被匹配到。

		如果出现了不被匹配的子串，那么正则表达式可以自动将子串拆开匹配。

		比如`aab`，要求不出现连续的`a`，那么正则就应该匹配成`a`，`ab`。

	2. 在需要匹配的模式出现前，为模式的出现创造前提条件

		比如一个匹配要求不出现`abb`，那么`a`的前面必定是`b*`，所以正则的第一个位置填`b*`。

	3. 只写可以出现的，不写不可能出现的

	cached question:

	1. 为什么正则表达式无法应用以树为代表的分析方式？

		为什么不可以把正则表达式看作是一个树的展开？

### nfa

**非确定有限自动机（NFA, non-definite finite automata）**

非确定有限自动机是一种需要对从一个状态出发的多条标有相同符号的边进行选择的自动机。

比如对于初始状态$s_0$，它向外有 2 条边，每条边的条件都是字母`a`，由此可以得到两个完全不同的终止条件。

标有$\epsilon$的边可以在不接收输入字符的情况下进行状态转换。

**正则表达式与 nfa 的转换**

每一个正则表达式都可以很容易地转换成一个 nfa，因此 nfa 很有用。

只需要将一个 nfa 转换成一个 dfa，就可以完成词法解析的任务。

看起来正则表达式描述的是确定有限状态机中的边，如果有$|$，那么就一个节点画出 2 条边；如果有$\cdot$，那么就一个节点连下一个节点；如果有$*$，那么就是一个终止节点，并且自己指向自己。

将小的状态机合并成大的：从节点 1 向外引出多个 edge，分别对应每个小状态机的第一条 edge。

**将 nfa 转换成 dfa**

在使用 nfa 处理字符串时，我们需要猜测接下来执行哪条边。

有一些算法可以总是猜对执行哪条边，方法是遍历一遍字符串，看看局部的转换条件是否被包含在 nfa 中。

### bison and flex

lex 可以将正则表达式转换成执行相应的代码。

flex 节省了 dfa 需要存储的状态，从而可以运行得更快。

使用 bison 和 flex 创建一个简易计算器。

创建一个工程文件夹，在文件夹中创建一个新文件`calc.y`，写入以下内容：

`calc.y`:

```bison
%{
    #include <stdio.h>
    #include <assert.h>
    static int Pop();
    static int Top();
    static void Push(int val);
%}

%token T_Int

%%

S : S E '\n' { printf("= %d\n", Top()); }
  |
  ;
E : E E '+' { Push(Pop() + Pop()); }
  | E E '-' { int op2 = Pop(); Push(Pop() - op2); }
  | E E '*' { Push(Pop() * Pop()); }
  | E E '/' { int op2 = Pop(); Push(Pop() / op2); }
  | T_Int   { Push(yylval); }
  ;

%%

static int stack[100], count = 0;
static int Pop() {
    assert(count > 0);
    return stack[--count];
}
static int Top() {
    assert(count > 0);
    return stack[count-1];
}
static void Push(int val) {
    assert(count < sizeof(stack)/sizeof(*stack));
    stack[count++] = val;
}
int main() {
    return yyparse();
}
```

再创建一个`calc.l`，输入以下内容：

`calc.l`:

```lex
%{
    #include "y.tab.h"
%}
%%
[0-9]+   { yylval = atoi(yytext); return T_Int;}
[-+*/\n] { return yytext[0];}
.        { /* ignore everything else */ }
```

创建一个 makefile 文件：

`Makefile`:

```makefile
calc: lex.yy.o y.tab.o
	gcc -o calc lex.yy.o y.tab.o -ly -ll

lex.yy.c: calc.l y.tab.c
	flex calc.l

y.tab.c: calc.y
	bison -vdty calc.y

clean:
	rm -f lex.yy.c y.tab.c y.tab.h lex.yy.o y.tab.o calc
```

在生成解析器前，需要先安装`bison`和`flex`：

```bash
sudo apt install bison flex
```

最后执行 makefile 就可以了：

```bash
make
```

此时会生成一个`calc`文件，运行它：

```bash
./calc
```

然后输入：

```
1 2 +
```

按回车后，可以看到程序的输出为

```
= 3
```

表示编译器运行成功。

### 文法

#### 二义性

如果给定一套文法，根据一定的推导规则，推导出来了两套语法树，那么就认为这套文法有二义性。

消除二义性的方法是在文法中再引入一些非终结符，从而赋予不同的非终结符结合优先级。另外在推导规则上，可以总是遵循从左到右。

#### 终结符，非终结符，与推导

凡是写成$\text{left} \rightarrow \text{right}$的，都可以将左边的符号用右边替换。

终结符（terminal）：指 id, print, num, `.`, `+`, `()`, `:=`, `;`

终结符对应到具体单词或者运算符

非终结符（nonterminal）：指`S`，`E`，`L`这些

非终结符是为了分析使用的，创造的抽象符号。

## 课后习题

* 一些自己写的答案

    * 编译器习题 1.1 的一个实现

        ```cpp
        #include <iostream>
        #include <string.h>
        using namespace std;

        typedef const char* String;
        #define TRUE 1
        #define FALSE 0

        typedef struct tree *T_tree;
        struct tree {
            T_tree left;
            String key;
            T_tree right;
        };

        T_tree Tree(T_tree l, String k, T_tree r)
        {
            T_tree t = (T_tree) malloc(sizeof(*t));
            t->left = l;
            t->key = k;
            t->right = r;
            return t;
        }

        T_tree insert(String key, T_tree t)
        {
            if (t == NULL)
                return Tree(NULL, key, NULL);
            else if (strcmp(key, t->key) < 0)
                return Tree(insert(key, t->left), t->key, t->right);
            else if (strcmp(key, t->key) > 0)
                return Tree(t->left, t->key, insert(key, t->right));
            else
                return Tree(t->left, key, t->right);
        }

        bool member(String key, T_tree root)
        {
            if (root == NULL)
                return FALSE;

            if (strcmp(key, root->key) == 0)
                return TRUE;
            else if (strcmp(key, root->key) < 0)
                return member(key, root->left);
            else
                return member(key, root->right);
        }

        T_tree insert(String key, void *binding, T_tree t);
        void *lookup(String key, T_tree t);

        int main()
        {
            T_tree root = NULL;
            root = insert("hello", root);
            root = insert("world", root);
            root = insert("nihao", root);
            root = insert("zaijian", root);

            String key = "zaijian";
            bool found = member(key, root);
            if (found)
                printf("member %s exists.\n", key);
            else
                printf("member %s doesn't exist.\n", key);

            key = "haha";
            found = member(key, root);
            if (found)
                printf("member %s exists.\n", key);
            else
                printf("member %s doesn't exist.\n", key);

            return 0;
        }
        ```

        输出：

        ```
        member zaijian exists.
        member haha doesn't exist.
        ```

        这段代码，每次`insert()`，都会返回一个全新的树，所以习题上才说旧的树还可继续用于查找。

        这个`insert()`不会插入重复的元素，如果某个元素已经存在，那么就返回原树的一个副本。

        `typedef struct tree *T_tree;`实际上就是指定了一个新类型的指针，有些代码可能没有`struct tree`的定义，纯粹是为了区分类型。这样的操作常见于句柄。

        `T_tree t = (T_tree) malloc(sizeof(*t));`这种写法还是第一次见，看来等号左边写出来的变量，右边就可以直接用了。或者说，这个语句，其实是声明和赋值的结合体。

        b 小题没看懂。

    * 2.1 a

        $c^*b^+(c^*a^*b^*)^*a^+(a^*b^*c^*)^*$

    * 2.1 b

        如果以$a$开始，必须以$a$结束：$(a(b^*c^*)^*a)$

        如果不以$a$开始，那么中间字符串可以有上面的模式，也可以没有：$((b^*c^*)^*a(b^*c^*)^*a(b^*c^*)^*)^*$

    * 2.1 c

        4 的倍数要求最后三位是$100$，即$(1^*0^*)^*100$

    * 2.1 d

        情况 1：位数比 6 大，不考虑后面的位：$(1^+0^*)^+(1|0)^6$

        情况2：位数为 6，则对于`101001`从左往右各个位情况如下：

        1. 这位必须为 1

        2. 这位可以是 0 可以是 1

        3. 如果前面一位是 1，这位可为 0 可为 1；如果前面一位是 0，这位必为 1

        4. 可为 0 可为 1

        5. 0 或 1

        6. 若前面为 1，这里 0 或 1 都可以；若前面为 0 ，这里 0 或 1 都不可以

        由 5 和 6 联合推导出，第 5 位必为 1，第 6 位可为 0 可为 1

        由此得到：$1(1(1 | 0) | 01)(0 | 1)(1(0 | 1))$

        感觉这个不太对，还得再改改。因为如果前几位已经可以确定大小，后面的位数就可以不考虑了。

    * 2.2 a

        a 比 b 多的情况有无限种，所以无法匹配。
