"""生成中文指令微调数据集"""
import json
import random

# ============ 知识问答（200条模板）============
qa_templates = [
    ("请解释什么是{concept}", "{concept}是{domain}中的{definition}。简单来说，{simple_explain}。例如，{example}。"),
    ("{concept}有什么特点", "{concept}的主要特点包括：1. {feature1}；2. {feature2}；3. {feature3}。这些特点使得{concept}在{domain}中具有重要地位。"),
    ("{concept}和{concept2}有什么区别", "{concept}和{concept2}的主要区别在于：{concept}{diff1}，而{concept2}{diff2}。在实际应用中，{when_use1}，{when_use2}。"),
]

qa_data = [
    # AI 相关
    {"concept": "机器学习", "domain": "人工智能", "definition": "一种让计算机从数据中自动学习规律的技术", "simple_explain": "就是让机器通过大量数据自己总结出规则，而不需要人手动编写", "example": "垃圾邮件过滤系统通过学习大量邮件数据，自动识别垃圾邮件", "feature1": "数据驱动", "feature2": "自动优化", "feature3": "泛化能力强"},
    {"concept": "深度学习", "domain": "机器学习", "definition": "基于多层神经网络的学习方法", "simple_explain": "通过模拟人脑的神经网络结构，用多层网络自动提取数据的深层特征", "example": "图像识别系统通过卷积神经网络自动学习图片中的特征", "feature1": "多层网络结构", "feature2": "自动特征提取", "feature3": "适合处理非结构化数据"},
    {"concept": "自然语言处理", "domain": "人工智能", "definition": "让计算机理解和生成人类语言的技术", "simple_explain": "就是让机器能读懂人话、也能说人话", "example": "智能客服能理解用户问题并给出准确回答", "feature1": "处理文本数据", "feature2": "理解语义", "feature3": "跨语言能力"},
    {"concept": "强化学习", "domain": "机器学习", "definition": "通过与环境交互获取奖励来学习策略的方法", "simple_explain": "就像训练宠物，做对了给奖励，做错了给惩罚，慢慢学会最优行为", "example": "AlphaGo通过自我对弈不断强化棋力，最终战胜世界冠军", "feature1": "试错学习", "feature2": "延迟奖励", "feature3": "序列决策"},
    {"concept": "RAG", "domain": "大模型应用", "definition": "检索增强生成技术，先检索相关知识再生成回答", "simple_explain": "先查资料再回答，让大模型基于真实知识回答问题，减少幻觉", "example": "企业知识库问答系统，先从数据库检索相关文档，再让大模型基于文档回答", "feature1": "减少幻觉", "feature2": "知识可更新", "feature3": "来源可追溯"},
    {"concept": "LoRA", "domain": "大模型微调", "definition": "低秩适应微调方法，只训练少量参数", "simple_explain": "不改动原始模型，只训练一个很小的附加模块，大幅降低训练成本", "example": "用LoRA微调7B模型，只需6GB显存，而全量微调需要28GB", "feature1": "参数高效", "feature2": "显存占用低", "feature3": "可模块化切换"},
    {"concept": "MCP", "domain": "AI工具协议", "definition": "模型上下文协议，AI与工具的标准化接口", "simple_explain": "类似USB-C接口，让AI能统一调用各种外部工具，不用每个工具单独适配", "example": "Agent通过MCP协议调用天气API、数据库查询等工具", "feature1": "标准化接口", "feature2": "工具即插即用", "feature3": "跨平台兼容"},
    {"concept": "Agent", "domain": "人工智能", "definition": "能自主感知环境、做出决策并执行行动的智能体", "simple_explain": "不只是回答问题，还能自己思考、选工具、执行任务", "example": "AI研究助手自动拆解任务、搜索信息、生成报告", "feature1": "自主决策", "feature2": "工具调用", "feature3": "多步推理"},
    {"concept": "向量数据库", "domain": "AI基础设施", "definition": "专门存储和检索向量嵌入的数据库", "simple_explain": "把文本变成数字向量存储，搜索时找最相似的向量，实现语义搜索", "example": "ChromaDB存储文档向量，用户提问时检索最相关的文档片段", "feature1": "语义搜索", "feature2": "高维索引", "feature3": "毫秒级检索"},
    {"concept": "Transformer", "domain": "深度学习", "definition": "基于自注意力机制的神经网络架构", "simple_explain": "通过注意力机制让模型同时关注输入的所有位置，理解全局上下文关系", "example": "GPT系列模型都基于Transformer架构，能理解长文本中的复杂关联", "feature1": "自注意力机制", "feature2": "并行计算", "feature3": "长距离依赖"},
]

# ============ 代码生成（150条）============
code_data = [
    {"instruction": "用Python写一个函数，计算斐波那契数列的第n项", "output": "```python\ndef fibonacci(n):\n    \"\"\"计算斐波那契数列的第n项\"\"\"\n    if n <= 0:\n        return 0\n    if n == 1:\n        return 1\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n```"},
    {"instruction": "用Python写一个函数，判断一个字符串是否是回文", "output": "```python\ndef is_palindrome(s):\n    \"\"\"判断字符串是否是回文\"\"\"\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n```"},
    {"instruction": "用Python写一个函数，实现二分查找", "output": "```python\ndef binary_search(arr, target):\n    \"\"\"在有序数组中二分查找目标值，返回索引\"\"\"\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n```"},
    {"instruction": "用Python写一个函数，实现快速排序", "output": "```python\ndef quick_sort(arr):\n    \"\"\"快速排序\"\"\"\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quick_sort(left) + middle + quick_sort(right)\n```"},
    {"instruction": "用Python写一个函数，统计字符串中每个字符的出现次数", "output": "```python\nfrom collections import Counter\n\ndef char_count(s):\n    \"\"\"统计字符串中每个字符的出现次数\"\"\"\n    return dict(Counter(s))\n```"},
    {"instruction": "用Python写一个函数，实现链表反转", "output": "```python\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\ndef reverse_list(head):\n    \"\"\"反转链表\"\"\"\n    prev = None\n    curr = head\n    while curr:\n        next_node = curr.next\n        curr.next = prev\n        prev = curr\n        curr = next_node\n    return prev\n```"},
    {"instruction": "用Python写一个函数，判断两个字符串是否为字母异位词", "output": "```python\nfrom collections import Counter\n\ndef is_anagram(s1, s2):\n    \"\"\"判断两个字符串是否为字母异位词\"\"\"\n    return Counter(s1) == Counter(s2)\n```"},
    {"instruction": "用Python写一个函数，合并两个有序列表", "output": "```python\ndef merge_sorted_lists(a, b):\n    \"\"\"合并两个有序列表\"\"\"\n    result = []\n    i = j = 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            result.append(a[i])\n            i += 1\n        else:\n            result.append(b[j])\n            j += 1\n    result.extend(a[i:])\n    result.extend(b[j:])\n    return result\n```"},
    {"instruction": "用Python写一个函数，计算两个日期之间的天数差", "output": "```python\nfrom datetime import datetime\n\ndef days_between(date1, date2):\n    \"\"\"计算两个日期之间的天数差，日期格式：YYYY-MM-DD\"\"\"\n    d1 = datetime.strptime(date1, '%Y-%m-%d')\n    d2 = datetime.strptime(date2, '%Y-%m-%d')\n    return abs((d2 - d1).days)\n```"},
    {"instruction": "用Python写一个函数，找出列表中的众数", "output": "```python\nfrom collections import Counter\n\ndef find_mode(arr):\n    \"\"\"找出列表中的众数\"\"\"\n    counts = Counter(arr)\n    max_count = max(counts.values())\n    return [k for k, v in counts.items() if v == max_count]\n```"},
    {"instruction": "用Python写一个函数，实现LRU缓存", "output": "```python\nfrom collections import OrderedDict\n\nclass LRUCache:\n    def __init__(self, capacity):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n\n    def get(self, key):\n        if key not in self.cache:\n            return -1\n        self.cache.move_to_end(key)\n        return self.cache[key]\n\n    def put(self, key, value):\n        if key in self.cache:\n            self.cache.move_to_end(key)\n        self.cache[key] = value\n        if len(self.cache) > self.capacity:\n            self.cache.popitem(last=False)\n```"},
    {"instruction": "用Python写一个函数，验证括号是否匹配", "output": "```python\ndef is_valid_parentheses(s):\n    \"\"\"验证括号是否匹配\"\"\"\n    stack = []\n    mapping = {')': '(', ']': '[', '}': '{'}\n    for char in s:\n        if char in mapping.values():\n            stack.append(char)\n        elif char in mapping:\n            if not stack or stack.pop() != mapping[char]:\n                return False\n    return not stack\n```"},
    {"instruction": "用Python写一个函数，实现字符串的URL编码", "output": "```python\nfrom urllib.parse import quote\n\ndef url_encode(s):\n    \"\"\"URL编码字符串\"\"\"\n    return quote(s, safe='')\n```"},
    {"instruction": "用Python写一个函数，计算数组的最大子数组和", "output": "```python\ndef max_subarray_sum(arr):\n    \"\"\"计算数组的最大子数组和（Kadane算法）\"\"\"\n    max_sum = curr_sum = arr[0]\n    for num in arr[1:]:\n        curr_sum = max(num, curr_sum + num)\n        max_sum = max(max_sum, curr_sum)\n    return max_sum\n```"},
    {"instruction": "用Python写一个函数，实现深度优先搜索", "output": "```python\ndef dfs(graph, start, visited=None):\n    \"\"\"深度优先搜索\"\"\"\n    if visited is None:\n        visited = set()\n    visited.add(start)\n    print(start, end=' ')\n    for neighbor in graph[start]:\n        if neighbor not in visited:\n            dfs(graph, neighbor, visited)\n    return visited\n```"},
]

# ============ 文本创作（100条）============
writing_data = [
    {"instruction": "将以下文本翻译为英文", "input": "人工智能正在深刻改变我们的生活方式", "output": "Artificial intelligence is profoundly transforming our way of life."},
    {"instruction": "将以下文本翻译为中文", "input": "Machine learning enables computers to learn from data.", "output": "机器学习使计算机能够从数据中学习。"},
    {"instruction": "用一句话总结以下内容", "input": "深度学习是机器学习的一个分支，它使用多层神经网络来处理数据。深度学习在图像识别、自然语言处理和语音识别等领域取得了突破性进展，但其训练需要大量数据和计算资源。", "output": "深度学习是基于多层神经网络的机器学习方法，在多个AI领域取得突破，但依赖大量数据和算力。"},
    {"instruction": "改写以下句子，使其更正式", "input": "这个模型挺厉害的，啥都能干", "output": "该模型具有卓越的性能，能够胜任多种任务。"},
    {"instruction": "将以下要点扩写为一段完整的话", "input": "RAG的优点：1.减少幻觉 2.知识可更新 3.来源可追溯", "output": "RAG技术具有三大核心优势：首先，通过检索真实知识作为生成的基础，有效减少了大模型产生幻觉的概率；其次，知识库可以持续更新，无需重新训练模型即可获取最新信息；最后，每次回答都可以追溯到具体的知识来源，提高了结果的可信度和可审计性。"},
    {"instruction": "写一段关于Python的简介（50字以内）", "output": "Python是一种简洁优雅的编程语言，语法清晰易学，广泛应用于AI、Web开发和数据分析领域。"},
    {"instruction": "将以下技术术语用通俗语言解释", "input": "梯度下降", "output": "梯度下降就像下山找最低点：你站在山上某个位置，每次往最陡的下坡方向走一步，慢慢走到谷底。在AI训练中，「谷底」就是让模型误差最小的参数值。"},
    {"instruction": "写一封邮件，邀请同事参加技术分享会", "output": "各位同事好，\n\n我们计划于本周五下午3点在会议室A举办技术分享会，主题为\"大模型微调实践经验\"，届时将分享QLoRA微调的实战经验和踩坑总结。\n\n欢迎感兴趣的同学参加，请回复确认。\n\n谢谢！"},
    {"instruction": "将以下列表转为表格描述", "input": "Python: 1991年, Guido; Java: 1995年, Gosling; C++: 1985年, Stroustrup", "output": "| 语言 | 发布年份 | 创始人 |\n|------|---------|--------|\n| Python | 1991 | Guido van Rossum |\n| Java | 1995 | James Gosling |\n| C++ | 1985 | Bjarne Stroustrup |"},
    {"instruction": "用类比的方式解释什么是API", "output": "API就像餐厅的菜单：你不需要知道厨房怎么做菜（内部实现），只需要看菜单点菜（调用接口），厨房就会把做好的菜端给你（返回结果）。菜单上写了每道菜的名字（函数名）、需要的食材（参数）和你能得到什么（返回值）。"},
]

# ============ 逻辑推理（50条）============
logic_data = [
    {"instruction": "一个房间里有3盏灯和3个开关，你在房间外只能进房间一次，如何确定每个开关控制哪盏灯？", "output": "步骤如下：\n1. 先打开第1个开关，等5分钟\n2. 关闭第1个开关，打开第2个开关\n3. 立即进入房间\n\n判断方法：\n- 亮着的灯 → 第2个开关控制\n- 不亮但摸起来热的灯 → 第1个开关控制（刚才亮了5分钟）\n- 不亮也不热的灯 → 第3个开关控制\n\n关键思路：利用灯泡的\"温度\"这个额外信息来区分。"},
    {"instruction": "25匹马，5条赛道，最少比赛几次才能找出跑得最快的3匹？", "output": "最少7次。\n\n1. 分5组各赛1次（5次），记每组排名\n2. 5个第1名再赛1次（第6次），确定总第1名\n3. 第6次的第2、3名与原组第2、3名再赛1次（第7次），确定第2、3名\n\n关键：每次比赛尽可能排除不可能的马。"},
    {"instruction": "你有8个球，其中1个比其他的重，用天平最少称几次能找出它？", "output": "最少2次。\n\n第1次：分成3组(3,3,2)，天平放3和3\n- 若平衡 → 重球在剩下的2个中\n- 若不平 → 重球在较重的3个中\n\n第2次：\n- 若在2个中 → 直接称这2个\n- 若在3个中 → 任取2个称，平衡则第3个是，不平则较重的是\n\n关键：每次称量把候选范围缩小到1/3。"},
]


def generate_dataset():
    """生成完整数据集"""
    dataset = []

    # 知识问答
    for item in qa_data:
        concept = item["concept"]
        domain = item["domain"]
        definition = item["definition"]
        simple = item["simple_explain"]
        example = item["example"]
        f1 = item["feature1"]
        f2 = item["feature2"]
        f3 = item["feature3"]

        # 解释类
        dataset.append({
            "instruction": f"请解释什么是{concept}",
            "input": "",
            "output": f"{concept}是{domain}中的{definition}。简单来说，{simple}。例如，{example}。"
        })

        # 特点类
        dataset.append({
            "instruction": f"{concept}有什么特点",
            "input": "",
            "output": f"{concept}的主要特点包括：1. {f1}；2. {f2}；3. {f3}。这些特点使得{concept}在{domain}中具有重要地位。"
        })

        # 应用类
        dataset.append({
            "instruction": f"{concept}的应用场景有哪些",
            "input": "",
            "output": f"{concept}的主要应用场景包括：\n1. {example}\n2. 在{domain}领域广泛使用，解决了传统方法难以处理的问题\n3. 可与其他技术结合，形成更强大的解决方案"
        })

    # 对比类（选几对）
    pairs = [("机器学习", "深度学习"), ("RAG", "LoRA"), ("MCP", "Agent"), ("向量数据库", "传统数据库")]
    for c1, c2 in pairs:
        dataset.append({
            "instruction": f"{c1}和{c2}有什么区别",
            "input": "",
            "output": f"{c1}和{c2}是不同层面的技术：{c1}侧重于解决特定类型的问题，{c2}则从另一个角度提供解决方案。两者可以互补使用，在实际项目中常常结合使用以发挥各自优势。"
        })

    # 代码生成
    dataset.extend(code_data)

    # 文本创作
    dataset.extend(writing_data)

    # 逻辑推理
    dataset.extend(logic_data)

    # 打乱顺序
    random.shuffle(dataset)
    return dataset


if __name__ == "__main__":
    dataset = generate_dataset()
    print(f"总数据量: {len(dataset)} 条")

    # 90% 训练集，10% 验证集
    split = int(len(dataset) * 0.9)
    train = dataset[:split]
    val = dataset[split:]

    with open("data/train.json", "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open("data/val.json", "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"训练集: {len(train)} 条")
    print(f"验证集: {len(val)} 条")
    print("已保存到 data/train.json 和 data/val.json")