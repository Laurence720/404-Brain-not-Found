"""
quiz_tool.py — Quiz Card generator

Features
--------
- Create 2–3 quiz questions (single-choice or true/false) around a finance term, each with a concise explanation.
- Optional RAG: when teach_db/ (FAISS + sentence-transformers) is available, the tool retrieves supporting context.
- Optional LLM (e.g., IBM watsonx); without an LLM it returns a deterministic “skeleton” quiz.
- Provides both a Python API and a LangChain StructuredTool for agent integration.

Return schema
-------------
{
  "card": {
    "term": "...",
    "questions": [
      {
        "type": "single" | "bool",
        "question": "...",
        "options": ["A","B","C","D"],     # for single-choice questions
        "answer": 0 | true | false,       # index for single-choice, boolean for true/false
        "explanation": "Short rationale (≤40 chars / ≤30 words)"
      }
    ],
    "citations": [{"title":"", "url":"", "source":""}],
    "lang": "zh" | "en"
  },
  "meta": {"used_rag": bool, "retrieved": int, "note"?: str}
}
"""

from __future__ import annotations
import os, json, random
import time
from typing import Any, Dict, List, Optional

# ---------- Optional dependencies (graceful degradation) ----------
try:
    from langchain_ibm import ChatWatsonx  # type: ignore
    _WATSONX_AVAILABLE = True
except Exception:
    ChatWatsonx = None  # type: ignore
    _WATSONX_AVAILABLE = False

try:
    from langchain_core.messages import SystemMessage, HumanMessage
    from langchain_core.tools import StructuredTool
    _LC_CORE_AVAILABLE = True
except Exception:
    SystemMessage = HumanMessage = StructuredTool = None  # type: ignore
    _LC_CORE_AVAILABLE = False

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    from langchain_community.vectorstores import FAISS  # type: ignore
    _RAG_AVAILABLE = True
except Exception:
    HuggingFaceEmbeddings = FAISS = None  # type: ignore
    _RAG_AVAILABLE = False


# ---------- LLM builder (optional watsonx) ----------
def _get_project_id() -> Optional[str]:
    for key in ("PROJ_ID", "PROJECT_ID", "WATSONX_PROJECT_ID"):
        v = os.getenv(key)
        if v:
            return v
    return None

def build_default_llm():
    if not _WATSONX_AVAILABLE:
        return None
    model_id = os.getenv("WATSONX_CHAT_MODEL", "ibm/granite-3-2-8b-instruct")
    project_id = _get_project_id()
    if not project_id:
        return None
    try:
        llm = ChatWatsonx(
            model_id=model_id,
            project_id=project_id,
            params={"decoding_method": "greedy", "max_new_tokens": 400, "temperature": 0.0},
        )
        return llm
    except Exception:
        return None


# ---------- teach_db retrieval (optional RAG) ----------
def _load_teach_vs(teach_db_dir: Optional[str] = None):
    base = teach_db_dir or os.path.join(os.path.dirname(__file__), "teach_db")
    
    if not _RAG_AVAILABLE:
        return None, False
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vs = FAISS.load_local(base, embeddings, allow_dangerous_deserialization=True)
        return vs.as_retriever(search_kwargs={"k": 3}), True
    except Exception:
        return None, False


def _format_docs_for_prompt(docs: List[Any], max_chars: int = 2800) -> str:
    if not docs:
        return ""
    chunks = []
    for doc in docs:
        text = getattr(doc, "page_content", str(doc))
        if text:
            chunks.append(text)
    combined = "\n\n".join(chunks)
    return combined[:max_chars] + ("..." if len(combined) > max_chars else "")


def _collect_citations(docs: List[Any], limit: int = 5) -> List[Dict[str, str]]:
    citations = []
    for doc in docs[:limit]:
        metadata = getattr(doc, "metadata", {})
        citations.append({
            "title": metadata.get("title", ""),
            "url": metadata.get("url", ""),
            "source": metadata.get("source", "")
        })
    return citations


# ---------- Educational helpers ----------
def _ensure_educational_explanation(explanation: Optional[str], term: str, lang: str) -> str:
    term_clean = str(term or "").strip()
    lang = (lang or "zh").lower()
    text = (str(explanation or "").strip())

    if lang.startswith("zh"):
        default_text = f"{term_clean} clarifies its role in investment decisions and risk awareness."
        if not text or len(text) < 10:
            text = default_text
        if term_clean and term_clean.lower() not in text.lower():
            text = f"{text} Focuses on the investment value of {term_clean}."
        if len(text) > 40:
            text = text[:40]
        return text or f"{term_clean} adds practical investment insight."

    # default to English
    default_en = f"{term_clean} helps investors understand its impact on portfolio risk and returns."
    if not text or len(text.split()) < 4:
        text = default_en
    if term_clean and term_clean.lower() not in text.lower():
        text = f"{text} Highlighting why {term_clean} matters to investors."
    words = text.split()
    if len(words) > 30:
        text = " ".join(words[:30])
    if not text:
        text = default_en
    return text


# ---------- JSON 抽取 ----------
def _extract_json(text: str, term: str = "quiz", lang: str = "zh") -> Optional[Dict[str, Any]]:
    import re
    try:
        # 首先尝试提取 ```json 代码块
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text)
        if m:
            json_str = m.group(1)
            result = json.loads(json_str)
            return result
        
        # 尝试提取 JSON 数组格式
        m = re.search(r"```json\s*(\[[\s\S]*?\])\s*```", text)
        if m:
            json_str = m.group(1)
            questions_array = json.loads(json_str)
            if isinstance(questions_array, list) and len(questions_array) > 0:
                # 修复答案格式和清理问题文本
                for q in questions_array:
                    # 清理问题文本
                    if "question" in q and isinstance(q["question"], str):
                        question_text = q["question"]
                        # 移除 "Answer: ..." 和 "Why: ..." 部分
                        question_text = re.sub(r'\s*Answer:\s*[^.]*\.?\s*', ' ', question_text)
                        question_text = re.sub(r'\s*Why:\s*[^.]*\.?\s*', ' ', question_text)
                        # 移除多余的数字编号（如 "3. "）
                        question_text = re.sub(r'^\d+\.\s*', '', question_text)
                        # 清理多余的空格
                        question_text = re.sub(r'\s+', ' ', question_text).strip()
                        # 确保问题以问号结尾
                        if not question_text.endswith('?'):
                            question_text += '?'
                        q["question"] = question_text
                    
                    # 修复答案格式
                    if q.get("type") == "single" and isinstance(q.get("answer"), str):
                        if q["answer"] in ["A", "B", "C", "D"]:
                            q["answer"] = ord(q["answer"]) - ord("A")
                    elif q.get("type") == "bool" and isinstance(q.get("answer"), str):
                        q["answer"] = q["answer"].lower() in ["true", "1", "yes"]
                    
                    # 确保有 explanation 字段并强化教育意义
                    if "explanation" not in q or not q["explanation"]:
                        q["explanation"] = ""
                    q["explanation"] = _ensure_educational_explanation(q["explanation"], term, lang)
                
                # 转换为期望的格式
                result = {
                    "card": {
                        "term": term,
                        "questions": questions_array,
                        "citations": [],
                        "lang": "en"
                    }
                }
                return result
        
        # 然后尝试提取普通 JSON，但只取第一个完整的 JSON 对象
        m = re.search(r"(\{[^{}]*\"card\"[^{}]*\{[^{}]*\"questions\"[^{}]*\}[^{}]*\})", text)
        if m:
            json_str = m.group(1)
            result = json.loads(json_str)
            return result
        
        # 如果还是失败，尝试提取任何看起来像 JSON 的内容
        m = re.search(r"(\{[\s\S]*?\"questions\"[\s\S]*?\})", text)
        if m:
            json_str = m.group(1)
            result = json.loads(json_str)
            return result
        
        return None
    except Exception:
        return None


# ---------- 答案一致性检查函数 ----------
def _fix_answer_consistency(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    检查并修正答案与解释的一致性
    """
    fixed_questions = []
    for q in questions:
        if q.get("type") == "single" and "options" in q and "explanation" in q:
            explanation = q["explanation"].lower()
            options = q["options"]
            current_answer = q["answer"]
            
            # 1. 检查解释中是否包含否定词（Yes/No 问题）
            has_negative = any(word in explanation for word in ["no", "not", "doesn't", "don't", "false", "incorrect", "wrong"])
            
            if has_negative and current_answer == 0 and len(options) > 1:
                # 查找 "No" 选项
                no_index = None
                for i, option in enumerate(options):
                    if option.lower().strip() in ["no", "false", "incorrect"]:
                        no_index = i
                        break
                
                if no_index is not None:
                    print(f"[DEBUG] Fixing Yes/No consistency: {q.get('question')}")
                    print(f"[DEBUG] Original answer: {current_answer} ({options[current_answer]})")
                    print(f"[DEBUG] Explanation: {explanation}")
                    print(f"[DEBUG] New answer: {no_index} ({options[no_index]})")
                    
                    q["answer"] = no_index
            elif not has_negative and current_answer == 1 and len(options) > 1:
                # 查找 "Yes" 选项
                yes_index = None
                for i, option in enumerate(options):
                    if option.lower().strip() in ["yes", "true", "correct"]:
                        yes_index = i
                        break
                
                if yes_index is not None:
                    print(f"[DEBUG] Fixing Yes/No consistency: {q.get('question')}")
                    print(f"[DEBUG] Original answer: {current_answer} ({options[current_answer]})")
                    print(f"[DEBUG] Explanation: {explanation}")
                    print(f"[DEBUG] New answer: {yes_index} ({options[yes_index]})")
                    
                    q["answer"] = yes_index
            
            # 2. 检查解释中的关键概念与选项的匹配度
            else:
                # 提取解释中的关键概念
                explanation_concepts = []
                for option in options:
                    option_text = option.lower().strip()
                    # 移除选项前缀（如 "A. ", "B. "）
                    if option_text.startswith(('a. ', 'b. ', 'c. ', 'd. ')):
                        option_text = option_text[3:]
                    
                    # 检查解释中是否包含这个选项的关键词
                    option_words = option_text.split()
                    for word in option_words:
                        if len(word) > 3 and word in explanation:  # 只匹配长度>3的单词
                            explanation_concepts.append((option_text, options.index(option)))
                
                # 如果找到匹配的概念，且当前答案不匹配
                if explanation_concepts:
                    best_match = explanation_concepts[0]  # 取第一个匹配
                    best_option_text, best_index = best_match
                    
                    if best_index != current_answer:
                        print(f"[DEBUG] Fixing concept consistency: {q.get('question')}")
                        print(f"[DEBUG] Original answer: {current_answer} ({options[current_answer]})")
                        print(f"[DEBUG] Explanation: {explanation}")
                        print(f"[DEBUG] Best match: {best_index} ({options[best_index]})")
                        print(f"[DEBUG] Matched concept: {best_option_text}")
                        
                        q["answer"] = best_index
        
        fixed_questions.append(q)
    
    return fixed_questions


# ---------- 题目类型修正函数 ----------
def _fix_question_types(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    修正题目类型，确保选择疑问句使用单选题而不是判断题
    """
    fixed_questions = []
    for q in questions:
        question_text = q.get("question", "").lower()
        
        # 检查是否包含选择疑问句模式
        has_or_choice = (
            " or " in question_text or 
            "before or after" in question_text or
            "true or false" in question_text or
            "yes or no" in question_text
        )
        
        # 如果是判断题但包含选择疑问句，转换为单选题
        if q.get("type") == "bool" and has_or_choice:
            print(f"[DEBUG] Converting bool question to single: {q.get('question')}")
            
            # 根据问题内容生成合适的选项
            if "before or after" in question_text:
                options = ["Before", "After"]
                answer = 0 if q.get("answer") is True else 1
            elif "true or false" in question_text:
                options = ["True", "False"]
                answer = 0 if q.get("answer") is True else 1
            elif "yes or no" in question_text:
                options = ["Yes", "No"]
                answer = 0 if q.get("answer") is True else 1
            else:
                # 通用情况，从问题中提取选项
                if " or " in question_text:
                    parts = question_text.split(" or ")
                    if len(parts) >= 2:
                        options = [parts[0].strip().title(), parts[1].strip().title()]
                        answer = 0 if q.get("answer") is True else 1
                    else:
                        options = ["Option A", "Option B"]
                        answer = 0
                else:
                    options = ["Yes", "No"]
                    answer = 0 if q.get("answer") is True else 1
            
            # 创建新的单选题
            fixed_q = {
                "type": "single",
                "question": q.get("question"),
                "options": options,
                "answer": answer,
                "explanation": q.get("explanation", "Explanation unavailable")
            }
            fixed_questions.append(fixed_q)
        else:
            # 保持原题目不变
            fixed_questions.append(q)
    
    return fixed_questions


# ---------- 核心生成函数 ----------
def generate_quiz_card(
    term: str,
    language: str = "zh",
    max_questions: int = 3,
    prefer_types: List[str] = None,
    teach_db_dir: Optional[str] = None,
    llm = None,
    context_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    生成测验卡
    
    Args:
        term: 主题/术语
        language: 语言 ("zh" 或 "en")
        max_questions: 最大题目数
        prefer_types: 偏好类型 ["single", "bool"]
        teach_db_dir: 教学数据库目录
        llm: LLM 实例
    
    Returns:
        测验卡字典
    """
    if prefer_types is None:
        prefer_types = ["single", "bool"]
    
    # 加载 RAG 检索器
    retriever, rag_ok = _load_teach_vs(teach_db_dir)
    if rag_ok and retriever:
        docs = retriever.get_relevant_documents(term)
    else:
        docs = []
    
    if context_override:
        context = str(context_override)
    else:
        context = _format_docs_for_prompt(docs, max_chars=2800) if docs else ""
    citations = _collect_citations(docs, limit=5)

    # 构建 LLM
    chat = llm or build_default_llm()
    lang = (language or "zh").lower()

    output_lang_label = "Chinese" if lang.startswith("zh") else "English"
    lang_code = "zh" if lang.startswith("zh") else "en"
    sys_text = (
        "You are a finance education assistant. Every question must stay within finance, investing, capital markets, or corporate finance."
        "Produce a strict-JSON quiz card that obeys these rules:"
        "- Include 2–3 questions only; allowed types are \"single\" (four options, exactly one correct index 0-3) or \"bool\" (true/false)."
        "- If the prompt is a choice question (e.g., includes \"or\"), you must use the \"single\" type with explicit options; only clear yes/no prompts may use \"bool\"."
        "- Each explanation must be ≤40 characters/30 words and highlight why the concept matters for investment decisions, risk management, or portfolio construction."
        "- Generate the content in the requested output language and set the JSON \"lang\" field to \"zh\" or \"en\" accordingly."
        "- Before responding, validate that the JSON is well-formed (fields, data types, option counts, answer indices) and fix any issues."
        "- Do not fabricate facts or URLs. Return exactly one ```json``` code block using the schema below:"
        "```json\n{\n  \"card\": {\n    \"term\": \"...\",\n    \"questions\": [\n      {\n        \"type\": \"single\",\n        \"question\": \"...\",\n        \"options\": [\"A\",\"B\",\"C\",\"D\"],\n        \"answer\": 2,\n        \"explanation\": \"...\"\n      }\n    ],\n    \"citations\": [{\"title\": \"\", \"url\": \"\", \"source\": \"\"}],\n    \"lang\": \"zh\"\n  }\n}\n```"
    )
    human_text = (
        f"[TERM] {term}\n"
        f"[OUTPUT_LANGUAGE] {output_lang_label} (set JSON \"lang\" = \"{lang_code}\")\n"
        f"[CONTEXT]\n{context}\n"
        f"[MAX_QUESTIONS] {max_questions}\n"
        f"[PREFERRED_TYPES] {','.join(prefer_types)}\n"
        "Return exactly one ```json``` fenced block and nothing else."
    )

    if (chat is not None) and _LC_CORE_AVAILABLE:
        msgs = [SystemMessage(content=sys_text), HumanMessage(content=human_text)]
        try:
            ai = chat.invoke(msgs)
            raw = getattr(ai, "content", "") or ""
            print(f"[DEBUG] LLM raw response: {raw[:500]}...")
            
            parsed = _extract_json(raw, term, lang) or {}
            print(f"[DEBUG] Parsed JSON: {parsed}")
            if not parsed:
                raise Exception("JSON parsing failed")
            
            # 确保有 card 字段
            if "card" not in parsed:
                raise Exception("Missing card field")
            
            card = parsed["card"]
            if "questions" not in card:
                raise Exception("Missing questions field")
            
            # 确保每个题目都有必要字段，并修复答案格式
            for q in card["questions"]:
                if "type" not in q or "question" not in q:
                    raise Exception("Invalid question format")
                if q["type"] == "single" and "options" not in q:
                    raise Exception("Invalid single choice question: missing options")
                if q["type"] == "single" and "answer" not in q and "correct_index" not in q:
                    raise Exception("Invalid single choice question: missing answer/correct_index")
                if q["type"] == "bool" and "answer" not in q:
                    raise Exception("Invalid bool question")
                
                # 清理问题文本，移除多余的答案和解释
                question_text = q["question"]
                if isinstance(question_text, str):
                    # 移除 "Answer: ..." 和 "Why: ..." 部分
                    import re
                    # 移除 Answer: 和 Why: 及其后面的内容
                    question_text = re.sub(r'\s*Answer:\s*[^.]*\.?\s*', ' ', question_text)
                    question_text = re.sub(r'\s*Why:\s*[^.]*\.?\s*', ' ', question_text)
                    # 移除多余的数字编号（如 "3. "）
                    question_text = re.sub(r'^\d+\.\s*', '', question_text)
                    # 清理多余的空格
                    question_text = re.sub(r'\s+', ' ', question_text).strip()
                    # 确保问题以问号结尾
                    if not question_text.endswith('?'):
                        question_text += '?'
                    q["question"] = question_text
                
                # 修复答案格式
                if q["type"] == "single" and "answer" in q and isinstance(q["answer"], str):
                    if q["answer"] in ["A", "B", "C", "D"]:
                        q["answer"] = ord(q["answer"]) - ord("A")
                elif q["type"] == "single" and "answer" not in q and "correct_index" in q:
                    try:
                        q["answer"] = int(q.pop("correct_index"))
                    except Exception as exc:
                        raise Exception(f"Invalid correct_index format: {exc}")
                elif q["type"] == "bool" and isinstance(q["answer"], str):
                    q["answer"] = q["answer"].lower() in ["true", "1", "yes"]
                
                # 确保有 explanation 字段并强化教育意义
                if "explanation" not in q or not q["explanation"]:
                    q["explanation"] = ""
                q["explanation"] = _ensure_educational_explanation(q["explanation"], term, lang)
            
            # 修正题目类型，确保选择疑问句使用单选题
            card["questions"] = _fix_question_types(card["questions"])
            
            # 检查并修正答案与解释的一致性
            card["questions"] = _fix_answer_consistency(card["questions"])
            
            return {
                "card": card,
                "meta": {
                    "used_rag": rag_ok,
                    "retrieved": len(docs),
                    "note": "Generated with LLM"
                }
            }
        except Exception:
            pass  # 降级到骨架卡
    
    # 降级：返回骨架卡
    skeleton_questions = []
    for i in range(min(max_questions, 3)):
        q_type = random.choice(prefer_types) if prefer_types else "single"
        if q_type == "single":
            skeleton_questions.append({
                "type": "single",
                "question": f"In an investing context, multiple-choice question {i+1} about {term}?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "answer": 0,
                "explanation": _ensure_educational_explanation("", term, lang)
            })
        else:  # bool
            skeleton_questions.append({
                "type": "bool",
                "question": f"In finance, is statement {i+1} about {term} correct?",
                "answer": True,
                "explanation": _ensure_educational_explanation("", term, lang)
            })
    
    return {
        "card": {
            "term": term,
            "questions": skeleton_questions,
            "citations": citations,
            "lang": lang
        },
        "meta": {
            "used_rag": rag_ok,
            "retrieved": len(docs),
            "note": "Skeleton card (no LLM available)"
        }
    }


# ---------- LangChain 工具包装 ----------
def get_quiz_tool() -> Optional[StructuredTool]:
    """返回 LangChain StructuredTool"""
    if not _LC_CORE_AVAILABLE:
        return None
    
    def _quiz_tool(term: str, language: str = "zh", max_questions: int = 3) -> str:
        """生成测验卡工具"""
        result = generate_quiz_card(term, language, max_questions)
        return json.dumps(result, ensure_ascii=False, indent=2)
    
    return StructuredTool.from_function(
        func=_quiz_tool,
        name="quiz_tool",
        description="生成投资主题的测验卡，包含2-3道单选或判断题",
        args_schema=None
    )


# ---------- 主程序（测试用） ----------
if __name__ == "__main__":
    # 测试
    result = generate_quiz_card("股票", "zh", 3)
    print(json.dumps(result, ensure_ascii=False, indent=2))
