"""
LangChain ReAct Agent - Flask Web Server V2
支持多用户 + 账号系统 + 独立记忆库
端口: 5000
"""

import os
import json
import re
import hashlib
import uuid
from datetime import datetime, timedelta
from functools import wraps
from typing import List, Dict, Optional

from flask import Flask, render_template, request, jsonify, session

# LangChain 核心组件
from langchain.agents import create_tool_calling_agent, AgentExecutor, Tool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 记忆管理器
from memory_manager_zhipu_v2 import get_memory_manager, MemoryManager


# ========================================
# 1. Flask 应用初始化
# ========================================

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this-in-production'


ZHIPU_API_KEY = "2fb4f7e613b14a8c8d0ffefd04bbcf0d.W0UXiNrd8LeWdexp"
ZHIPU_MODEL = "glm-4.6v"
ZHIPU_URL = "https://open.bigmodel.cn/api/paas/v4/"
DOUBAO_API_KEY = "7240d4cd-7258-4536-a5cc-afe23a21d5f4"
# 数据存储路径
DATA_DIR = "./familyAi/Langchainwithmemory/user_data"
os.makedirs(DATA_DIR, exist_ok=True)

USERS_FILE = os.path.join(DATA_DIR, "users.json")


# ========================================
# 2. 用户管理系统
# ========================================

def load_users() -> Dict:
    """加载用户数据"""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def save_users(users: Dict):
    """保存用户数据"""
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def hash_password(password: str) -> str:
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username: str, password: str) -> Dict:
    """创建新用户"""
    users = load_users()
    
    if username in users:
        return None
    
    user_id = str(uuid.uuid4())[:8]
    users[username] = {
        "user_id": user_id,
        "username": username,
        "password": password,
        "password_hash": hash_password(password),
        "created_at": datetime.now().isoformat(),
        "last_login": None
    }
    
    save_users(users)
    
    # 为用户创建独立的记忆库存储目录
    user_storage_dir = os.path.join(DATA_DIR, f"user_{user_id}")
    os.makedirs(user_storage_dir, exist_ok=True)
    
    return users[username]


def verify_user(username: str, password: str) -> Dict:
    """验证用户登录"""
    users = load_users()
    
    if username not in users:
        return None
    
    user = users[username]
    if user["password_hash"] != hash_password(password):
        return None
    
    # 更新最后登录时间
    user["last_login"] = datetime.now().isoformat()
    save_users(users)
    
    return user


# ========================================
# 3. 登录验证装饰器
# ========================================

def login_required(f):
    """需要登录的装饰器"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'success': False, 'error': '请先登录'}), 401
        return f(*args, **kwargs)
    return decorated_function


# ========================================
# 4. 使用 ChatOpenAI 配置智谱AI
# ========================================

llm = ChatOpenAI(
    model=ZHIPU_MODEL,
    api_key=ZHIPU_API_KEY,
    base_url=ZHIPU_URL,
    temperature=0.2
)


# ========================================
# 5. 定义工具（与用户无关的工具函数）
# ========================================

def add_memory_func(content: str, user_id: str = None) -> str:
    """工具: 添加记忆（带用户ID）"""
    try:
        if " | " in content:
            parts = content.split(" | ", 1)
            date_label = parts[0].strip()
            actual_content = parts[1].strip()
        else:
            date_label = datetime.now().strftime("%Y年%m月%d日")
            actual_content = content.strip()
        
        # 获取当前用户的记忆管理器
        mm = get_memory_manager(
            zhipu_api_key=ZHIPU_API_KEY,
            embedding_model="embedding-3",
            user_id=user_id  # 传入用户ID
        )
        
        formatted = mm.format_memory_template(content=actual_content)
        memory_id = mm.add_memory(
            content=actual_content,
            category=formatted['category'],
            tags=formatted['tags']
        )
        
        mm.memories[memory_id]['date_label'] = date_label
        mm._save_memories()
        
        return f"记忆添加成功！ID: {memory_id} | 时间: {date_label} | 内容: {actual_content[:50]}"
        
    except Exception as e:
        return f"添加失败: {str(e)}"


def query_memory_func(query: str, user_id: str = None) -> str:
    """工具: 查询记忆（带用户ID）"""
    try:
        mm = get_memory_manager(
            zhipu_api_key=ZHIPU_API_KEY,
            embedding_model="embedding-3",
            user_id=user_id
        )
        results = mm.query_memory(query, top_k=20)
        
        if not results:
            return "未找到相关记忆"
        
        response_parts = [f"找到 {len(results)} 条相关记忆："]
        
        for i, mem in enumerate(results, 1):
            sim = mem.get('similarity_score', 0)
            date_label = mem.get('date_label', '未知时间')
            memory_id = mem.get('id', '未知ID')
            response_parts.append(
                f"\n【记忆{i}】ID: {memory_id} | 内容: {mem['content'][:50]}... | 时间: {date_label} | 相关度: {sim:.1%}"
            )
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"查询失败: {str(e)}"


def delete_memory_func(memory_id: str, user_id: str = None) -> str:
    """工具: 删除记忆（带用户ID）"""
    try:
        original_input = memory_id
        cleaned_id = re.sub(r'[^a-zA-Z0-9]', '', memory_id)
        
        if len(cleaned_id) > 8:
            cleaned_id = cleaned_id[:8]
        
        if not cleaned_id or len(cleaned_id) != 8:
            return f"无效的记忆ID，请提供8位字母数字组合的ID"
        
        mm = get_memory_manager(
            zhipu_api_key=ZHIPU_API_KEY,
            embedding_model="embedding-3",
            user_id=user_id
        )
        success = mm.delete_memory(cleaned_id)
        
        if success:
            return f"记忆 {cleaned_id} 已删除"
        else:
            return f"未找到记忆 {cleaned_id}"
        
    except Exception as e:
        return f"删除失败: {str(e)}"


def query_memory_by_time_func(time_query: str, user_id: str = None) -> str:
    """工具: 按时间查询记忆（带用户ID）"""
    try:
        mm = get_memory_manager(
            zhipu_api_key=ZHIPU_API_KEY,
            embedding_model="embedding-3",
            user_id=user_id
        )
        results = mm.query_memory_by_time(time_query)
        
        if not results:
            return f"未找到 {time_query} 的记忆"
        
        response_parts = [f"找到 {len(results)} 条 {time_query} 的记忆："]
        
        for i, mem in enumerate(results, 1):
            date_label = mem.get('date_label', '未知时间')
            memory_id = mem.get('id', '未知ID')
            response_parts.append(
                f"\n【记忆{i}】ID: {memory_id} | 时间: {date_label} | 内容: {mem['content'][:50]}..."
            )
        
        return "\n".join(response_parts)
        
    except Exception as e:
        return f"按时间查询失败: {str(e)}"


# 创建工具包装函数（在运行时注入 user_id）
def create_tools_for_user(user_id: str):
    """为指定用户创建工具实例"""
    return [
        Tool(
            name="AddMemory",
            func=lambda x: add_memory_func(x, user_id),
            description="添加新的记忆内容。输入必须是'YYYY年MM月DD日 | 具体内容'的格式。例如：'2026年03月07日 | 我今天想去图书馆'"
        ),
        Tool(
            name="QueryMemory",
            func=lambda x: query_memory_func(x, user_id),
            description="""查询已有的记忆内容（语义搜索）。
适用于：根据内容描述查找相关记忆，如"我昨天吃了什么"、"关于AI的记忆"。
返回包含记忆ID、内容、时间和相关度的结果。
重要：返回的记忆ID可以用于 DeleteMemory 工具删除记忆。"""
        ),
        Tool(
            name="QueryMemoryByTime",
            func=lambda x: query_memory_by_time_func(x, user_id),
            description="""按时间查询记忆（精准时间搜索）。
适用于：用户明确指定时间的情况，如"3月14号的记忆"、"昨天的记录"。
重要：输入必须是具体的日期格式，如"2026年03月12日"或"2026-03-12"。
如果用户说"昨天"、"前天"、"今天"，请先计算具体日期再传入。
例如：用户问"前天做了什么"，今天是2026年03月14日，则输入"2026年03月12日"。
注意：如果用户没有明确指定时间，请使用 QueryMemory 进行语义搜索。"""
        ),
        Tool(
            name="DeleteMemory",
            func=lambda x: delete_memory_func(x, user_id),
            description="""删除指定的记忆。
输入：记忆ID（8位字母数字组合，如abc12345）。
注意：如果需要删除某条记忆但不知道ID，先使用 QueryMemory 或 QueryMemoryByTime 查询获取ID。"""
        )
    ]


# ========================================
# 6. 用户会话管理
# ========================================

# 存储每个用户的 Agent 实例
user_agents = {}


def get_agent_for_user(user_id: str):
    """为指定用户创建 Agent 实例"""
    if user_id not in user_agents:
        tools = create_tools_for_user(user_id)
        
        # Tool-calling Agent 的 Chat Prompt 模板
        react_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是一个帮助用户管理长期记忆的中文助手。

请遵守以下规则：
1. 结合对话历史理解用户的意图，特别是代词指代（如"那件事"、"那个"、"它"等）。
2. 只有在需要操作记忆（添加、查询、删除）时才调用工具。
3. 如果用户只是普通聊天，不要调用工具。
4. 如果用户要求记录一件事，只调用一次 AddMemory，除非用户明确要求记录多条不同内容。
5. 当工具已经返回足够结果后，不要重复调用同一个工具验证同一件事。

查询工具选择规则：
- QueryMemory：适用于用户描述内容但不指定具体时间的情况。
- QueryMemoryByTime：适用于用户明确指定日期的情况。
- 如果用户说"昨天"、"前天"、"今天"，先换算成具体日期再调用 QueryMemoryByTime。

删除记忆流程：
如果用户要删除某条记忆但没有提供 ID：
1. 先查询相关记忆
2. 从结果中提取 ID
3. 再调用 DeleteMemory

在给用户最终回复时，直接给出自然、简洁的中文答案，不要暴露推理过程。""",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        
        # 为每个用户创建独立的记忆（保留10轮对话）
        memory = ConversationBufferWindowMemory(
            k=10,
            memory_key="chat_history",
            return_messages=True
        )
        
        # 创建 Tool-calling Agent
        agent = create_tool_calling_agent(
            llm=llm,
            tools=tools,
            prompt=react_prompt
        )
        
        # 创建 Agent Executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=False,
            max_iterations=5
        )
        
        user_agents[user_id] = agent_executor
    
    return user_agents[user_id]


# ========================================
# 7. Flask 路由 - 用户认证
# ========================================

@app.route('/')
def index():
    """首页"""
    return render_template('index_v2.html')


@app.route('/api/register', methods=['POST'])
def register():
    """API: 用户注册"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'success': False, 'error': '用户名和密码不能为空'})
        
        if len(username) < 3 or len(password) < 6:
            return jsonify({'success': False, 'error': '用户名至少3位，密码至少6位'})
        
        user = create_user(username, password)
        if not user:
            return jsonify({'success': False, 'error': '用户名已存在'})
        
        return jsonify({
            'success': True,
            'message': '注册成功',
            'user_id': user['user_id'],
            'username': user['username']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/login', methods=['POST'])
def login():
    """API: 用户登录"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        password = data.get('password', '').strip()
        
        if not username or not password:
            return jsonify({'success': False, 'error': '用户名和密码不能为空'})
        
        user = verify_user(username, password)
        if not user:
            return jsonify({'success': False, 'error': '用户名或密码错误'})
        
        # 设置 session
        session['user_id'] = user['user_id']
        session['username'] = user['username']
        
        return jsonify({
            'success': True,
            'message': '登录成功',
            'user_id': user['user_id'],
            'username': user['username']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/logout', methods=['POST'])
def logout():
    """API: 用户登出"""
    session.clear()
    return jsonify({'success': True, 'message': '已登出'})


@app.route('/api/check_auth', methods=['GET'])
def check_auth():
    """API: 检查登录状态"""
    if 'user_id' in session:
        return jsonify({
            'success': True,
            'is_logged_in': True,
            'user_id': session['user_id'],
            'username': session['username']
        })
    return jsonify({'success': True, 'is_logged_in': False})


# ========================================
# 8. Flask 路由 - 聊天功能（需要登录）
# ========================================

@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    """API: 处理聊天请求"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        user_id = session['user_id']
        
        if not user_message:
            return jsonify({'success': False, 'error': '消息不能为空'})
        
        # 获取当前日期
        current_date = datetime.now().strftime("%Y年%m月%d日")
        enhanced_input = f"今天是{current_date}。{user_message}"
        
        # 获取该用户的 Agent
        agent_executor = get_agent_for_user(user_id)
        
        # 执行
        result = agent_executor.invoke({
            "input": enhanced_input
        })
        
        clean_output = result['output'].strip() if isinstance(result.get('output'), str) else result.get('output')
        
        return jsonify({
            'success': True,
            'response': clean_output
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/memories', methods=['GET'])
@login_required
def get_memories():
    """API: 获取当前用户的所有记忆"""
    try:
        user_id = session['user_id']
        mm = get_memory_manager(
            zhipu_api_key=ZHIPU_API_KEY,
            embedding_model="embedding-3",
            user_id=user_id
        )
        
        memories = []
        for mid, mem in mm.memories.items():
            if not mem.get('deleted', False):
                mem_copy = mem.copy()
                mem_copy['id'] = mid
                memories.append(mem_copy)
        
        return jsonify({
            'success': True,
            'count': len(memories),
            'memories': memories
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats():
    """API: 获取当前用户的统计信息"""
    try:
        user_id = session['user_id']
        mm = get_memory_manager(
            zhipu_api_key=ZHIPU_API_KEY,
            embedding_model="embedding-3",
            user_id=user_id
        )
        stats = mm.get_stats() if hasattr(mm, 'get_stats') else {}
        
        return jsonify({
            'success': True,
            'stats': stats,
            'current_date': datetime.now().strftime("%Y年%m月%d日")
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/clear_session', methods=['POST'])
@login_required
def clear_session():
    """API: 清除当前用户的对话上下文"""
    try:
        user_id = session['user_id']
        
        if user_id in user_agents:
            del user_agents[user_id]
        
        return jsonify({
            'success': True,
            'message': '对话上下文已清除'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


# ========================================
# 9. 启动
# ========================================

if __name__ == '__main__':
    current_date = datetime.now().strftime("%Y年%m月%d日")
    
    print("=" * 60)
    print("LangChain ReAct Agent - Flask Web Server V2")
    print("=" * 60)
    print(f"Current Date: {current_date}")
    print("Features: Multi-User + Auth + Independent Memory")
    print("-" * 60)
    print("Access URLs:")
    print("   Local:   http://127.0.0.1:5001")
    print("   Network: http://0.0.0.0:5001")
    print("=" * 60)
    print("Press Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=5001, debug=True)
