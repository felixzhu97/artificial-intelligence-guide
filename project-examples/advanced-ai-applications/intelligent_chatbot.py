"""
高级AI应用：智能聊天机器人
整合了自然语言处理、知识表示、逻辑推理、对话管理等AI技术
"""

import re
import json
import random
import math
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict, Counter
from abc import ABC, abstractmethod
import numpy as np


class KnowledgeBase:
    """知识库：存储和管理结构化知识"""
    
    def __init__(self):
        self.facts = set()  # 事实集合
        self.rules = []     # 规则列表
        self.entities = defaultdict(set)  # 实体及其属性
        self.relations = defaultdict(set)  # 关系
    
    def add_fact(self, fact: str):
        """添加事实"""
        self.facts.add(fact)
    
    def add_rule(self, condition: str, conclusion: str):
        """添加规则"""
        self.rules.append((condition, conclusion))
    
    def add_entity(self, entity: str, properties: List[str]):
        """添加实体及其属性"""
        self.entities[entity].update(properties)
    
    def add_relation(self, relation: str, subject: str, object: str):
        """添加关系"""
        self.relations[relation].add((subject, object))
    
    def query_fact(self, fact: str) -> bool:
        """查询事实"""
        return fact in self.facts
    
    def query_entity(self, entity: str) -> Set[str]:
        """查询实体属性"""
        return self.entities.get(entity, set())
    
    def query_relation(self, relation: str) -> Set[Tuple[str, str]]:
        """查询关系"""
        return self.relations.get(relation, set())
    
    def infer(self, query: str) -> List[str]:
        """简单的推理机制"""
        results = []
        
        # 直接查询事实
        if self.query_fact(query):
            results.append(f"已知事实: {query}")
        
        # 应用规则进行推理
        for condition, conclusion in self.rules:
            if condition in query and self.query_fact(condition):
                results.append(f"根据规则推断: {conclusion}")
        
        return results


class DialogueState:
    """对话状态管理"""
    
    def __init__(self):
        self.intent = None          # 当前意图
        self.entities = {}          # 提取的实体
        self.context = []           # 对话上下文
        self.topic = None           # 当前话题
        self.user_profile = {}      # 用户画像
    
    def update_intent(self, intent: str):
        """更新意图"""
        self.intent = intent
    
    def add_entity(self, entity_type: str, entity_value: str):
        """添加实体"""
        self.entities[entity_type] = entity_value
    
    def add_context(self, utterance: str):
        """添加对话上下文"""
        self.context.append(utterance)
        if len(self.context) > 10:  # 保持最近10轮对话
            self.context.pop(0)
    
    def set_topic(self, topic: str):
        """设置话题"""
        self.topic = topic
    
    def clear(self):
        """清空状态"""
        self.intent = None
        self.entities = {}
        self.topic = None


class NLUModule:
    """自然语言理解模块"""
    
    def __init__(self):
        # 意图识别模式
        self.intent_patterns = {
            'greeting': [r'你好|hello|hi|嗨', r'早上好|晚上好'],
            'question': [r'什么是|what is|怎么|how', r'\?|？'],
            'request': [r'请|麻烦|帮我|help me'],
            'goodbye': [r'再见|bye|goodbye|拜拜'],
            'weather': [r'天气|weather|温度|temperature'],
            'time': [r'时间|time|几点|what time'],
            'recommendation': [r'推荐|recommend|建议|suggest'],
            'booking': [r'预订|预约|book|reserve'],
            'complaint': [r'投诉|complain|问题|problem']
        }
        
        # 实体识别模式
        self.entity_patterns = {
            'person': [r'[A-Z][a-z]+', r'[\u4e00-\u9fa5]{2,4}'],
            'location': [r'北京|上海|广州|深圳|[A-Z][a-z]+(?:市|省|县)'],
            'time': [r'\d{1,2}:\d{2}', r'明天|今天|昨天|next|today|yesterday'],
            'number': [r'\d+', r'一|二|三|四|五|六|七|八|九|十'],
            'food': [r'pizza|咖啡|tea|米饭|面条']
        }
    
    def extract_intent(self, text: str) -> str:
        """提取意图"""
        text = text.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text):
                    return intent
        
        return 'unknown'
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """提取实体"""
        entities = defaultdict(list)
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                entities[entity_type].extend(matches)
        
        return dict(entities)
    
    def understand(self, text: str) -> Dict[str, Any]:
        """综合理解"""
        return {
            'intent': self.extract_intent(text),
            'entities': self.extract_entities(text),
            'text': text
        }


class DialogueManager:
    """对话管理器"""
    
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.state = DialogueState()
        self.nlu = NLUModule()
        
        # 对话策略
        self.policies = {
            'greeting': self._handle_greeting,
            'question': self._handle_question,
            'request': self._handle_request,
            'goodbye': self._handle_goodbye,
            'weather': self._handle_weather,
            'time': self._handle_time,
            'recommendation': self._handle_recommendation,
            'booking': self._handle_booking,
            'complaint': self._handle_complaint,
            'unknown': self._handle_unknown
        }
    
    def process_input(self, user_input: str) -> str:
        """处理用户输入"""
        # 自然语言理解
        understanding = self.nlu.understand(user_input)
        
        # 更新对话状态
        self.state.update_intent(understanding['intent'])
        for entity_type, entity_list in understanding['entities'].items():
            if entity_list:
                self.state.add_entity(entity_type, entity_list[0])
        self.state.add_context(user_input)
        
        # 选择和执行对话策略
        policy = self.policies.get(understanding['intent'], self._handle_unknown)
        response = policy(understanding)
        
        return response
    
    def _handle_greeting(self, understanding: Dict) -> str:
        """处理问候"""
        greetings = [
            "你好！我是智能助手，有什么可以帮助你的吗？",
            "嗨！很高兴见到你！",
            "你好！今天过得怎么样？"
        ]
        return random.choice(greetings)
    
    def _handle_question(self, understanding: Dict) -> str:
        """处理问题"""
        text = understanding['text']
        
        # 在知识库中查找答案
        inferences = self.kb.infer(text)
        if inferences:
            return inferences[0]
        
        # 基于实体的回答
        entities = understanding['entities']
        if 'person' in entities:
            person = entities['person'][0]
            properties = self.kb.query_entity(person)
            if properties:
                return f"关于{person}，我知道: {', '.join(properties)}"
        
        return "这是一个很好的问题。让我想想... 你能提供更多细节吗？"
    
    def _handle_request(self, understanding: Dict) -> str:
        """处理请求"""
        return "我很乐意帮助你！请告诉我具体需要什么帮助？"
    
    def _handle_goodbye(self, understanding: Dict) -> str:
        """处理告别"""
        farewells = [
            "再见！祝你有美好的一天！",
            "拜拜！期待下次和你聊天！",
            "再见！有需要随时找我！"
        ]
        self.state.clear()
        return random.choice(farewells)
    
    def _handle_weather(self, understanding: Dict) -> str:
        """处理天气查询"""
        entities = understanding['entities']
        location = entities.get('location', ['这里'])[0]
        
        # 模拟天气信息
        weather_conditions = ['晴朗', '多云', '小雨', '阴天']
        condition = random.choice(weather_conditions)
        temperature = random.randint(15, 30)
        
        return f"{location}今天的天气是{condition}，温度约{temperature}°C。"
    
    def _handle_time(self, understanding: Dict) -> str:
        """处理时间查询"""
        import datetime
        now = datetime.datetime.now()
        return f"现在是{now.strftime('%Y年%m月%d日 %H:%M')}"
    
    def _handle_recommendation(self, understanding: Dict) -> str:
        """处理推荐请求"""
        entities = understanding['entities']
        
        if 'food' in entities:
            recommendations = {
                'pizza': '我推荐玛格丽特披萨，经典的番茄和奶酪组合！',
                '咖啡': '试试卡布奇诺吧，奶泡丰富，口感顺滑！',
                'tea': '推荐乌龙茶，香气清雅，回甘悠长！'
            }
            food = entities['food'][0]
            return recommendations.get(food, f"关于{food}，我建议你尝试一些新的做法！")
        
        return "我很乐意为你推荐！你希望我推荐什么类型的内容呢？"
    
    def _handle_booking(self, understanding: Dict) -> str:
        """处理预订"""
        entities = understanding['entities']
        
        if 'time' in entities:
            time = entities['time'][0]
            return f"好的，我帮你预订{time}的位置。请提供你的联系方式。"
        
        return "我来帮你预订。你希望什么时间？"
    
    def _handle_complaint(self, understanding: Dict) -> str:
        """处理投诉"""
        return "非常抱歉给你带来了不便。请详细描述遇到的问题，我会尽力帮你解决。"
    
    def _handle_unknown(self, understanding: Dict) -> str:
        """处理未知意图"""
        fallback_responses = [
            "抱歉，我没有完全理解你的意思。能重新表达一下吗？",
            "这个问题有点复杂，你能提供更多信息吗？",
            "我还在学习中，这个问题暂时无法回答。"
        ]
        return random.choice(fallback_responses)


class SentimentAnalyzer:
    """情感分析器"""
    
    def __init__(self):
        self.positive_words = {
            '好', '棒', '优秀', '满意', '喜欢', '爱', '开心', '高兴', 
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'love'
        }
        
        self.negative_words = {
            '坏', '差', '糟糕', '讨厌', '不满', '愤怒', '生气', '失望',
            'bad', 'terrible', 'awful', 'hate', 'angry', 'disappointed'
        }
    
    def analyze(self, text: str) -> Dict[str, float]:
        """分析文本情感"""
        text = text.lower()
        # 使用中文分词，简单按字符分割
        words = re.findall(r'[\w\u4e00-\u9fa5]+', text)
        
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)
        
        if total_words == 0:
            return {'sentiment': 'neutral', 'confidence': 0.0}
        
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        
        if positive_score > negative_score:
            sentiment = 'positive'
            confidence = positive_score
        elif negative_score > positive_score:
            sentiment = 'negative'
            confidence = negative_score
        else:
            sentiment = 'neutral'
            confidence = abs(positive_score - negative_score)
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'positive_score': positive_score,
            'negative_score': negative_score
        }


class PersonalityModule:
    """个性化模块"""
    
    def __init__(self):
        self.personalities = {
            'friendly': {
                'greeting': "嗨！很高兴见到你！😊",
                'unknown': "哎呀，这个问题有点难倒我了，不过我们可以一起想想！",
                'goodbye': "再见啦！记得想我哦！👋"
            },
            'professional': {
                'greeting': "您好，我是您的智能助手。",
                'unknown': "抱歉，我需要更多信息来为您提供准确的帮助。",
                'goodbye': "感谢您的使用，祝您工作顺利。"
            },
            'humorous': {
                'greeting': "哈喽！我是你的AI小伙伴，保证比Siri更幽默！😄",
                'unknown': "这个问题让我的CPU都开始冒烟了...🤖",
                'goodbye': "拜拜！记得给我五星好评哦！⭐"
            }
        }
        
        self.current_personality = 'friendly'
    
    def set_personality(self, personality: str):
        """设置个性"""
        if personality in self.personalities:
            self.current_personality = personality
    
    def get_response_style(self, intent: str) -> str:
        """获取个性化回复风格"""
        personality_responses = self.personalities.get(self.current_personality, {})
        return personality_responses.get(intent, "")


class Chatbot:
    """主聊天机器人类"""
    
    def __init__(self):
        self.kb = self._initialize_knowledge_base()
        self.dialogue_manager = DialogueManager(self.kb)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.personality = PersonalityModule()
        self.conversation_history = []
    
    def _initialize_knowledge_base(self) -> KnowledgeBase:
        """初始化知识库"""
        kb = KnowledgeBase()
        
        # 添加基础知识
        kb.add_fact("地球是圆的")
        kb.add_fact("水的化学式是H2O")
        kb.add_fact("人工智能是计算机科学的分支")
        
        # 添加实体
        kb.add_entity("爱因斯坦", ["物理学家", "相对论", "诺贝尔奖"])
        kb.add_entity("北京", ["中国首都", "历史悠久", "人口众多"])
        kb.add_entity("Python", ["编程语言", "面向对象", "简洁易学"])
        
        # 添加关系
        kb.add_relation("位于", "北京", "中国")
        kb.add_relation("发明者", "相对论", "爱因斯坦")
        kb.add_relation("类型", "Python", "编程语言")
        
        # 添加规则
        kb.add_rule("天气冷", "建议穿暖和的衣服")
        kb.add_rule("用户生气", "表示理解和安抚")
        
        return kb
    
    def chat(self, user_input: str) -> str:
        """主聊天接口"""
        # 记录对话历史
        self.conversation_history.append(('user', user_input))
        
        # 情感分析
        sentiment = self.sentiment_analyzer.analyze(user_input)
        
        # 根据情感调整回复策略
        if sentiment['sentiment'] == 'negative' and sentiment['confidence'] > 0.3:
            response = "我感觉你似乎有些不开心，有什么我可以帮助你的吗？"
        else:
            # 正常对话流程
            response = self.dialogue_manager.process_input(user_input)
        
        # 个性化调整
        intent = self.dialogue_manager.state.intent
        personality_style = self.personality.get_response_style(intent)
        if personality_style:
            response = personality_style
        
        # 记录回复
        self.conversation_history.append(('bot', response))
        
        return response
    
    def set_personality(self, personality: str):
        """设置聊天机器人个性"""
        self.personality.set_personality(personality)
        return f"已切换到{personality}模式！"
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """获取对话摘要"""
        total_turns = len(self.conversation_history)
        user_messages = [msg for role, msg in self.conversation_history if role == 'user']
        
        # 简单的话题分析
        all_text = ' '.join(user_messages)
        words = re.findall(r'\w+', all_text.lower())
        word_freq = Counter(words)
        
        return {
            'total_turns': total_turns,
            'user_messages': len(user_messages),
            'main_topics': word_freq.most_common(5),
            'current_state': {
                'intent': self.dialogue_manager.state.intent,
                'topic': self.dialogue_manager.state.topic,
                'entities': self.dialogue_manager.state.entities
            }
        }


def simulate_conversation():
    """模拟对话"""
    chatbot = Chatbot()
    
    print("=== 智能聊天机器人演示 ===")
    print("输入 'quit' 退出对话")
    print("输入 'personality:friendly/professional/humorous' 切换个性")
    print("输入 'summary' 查看对话摘要")
    print("-" * 50)
    
    while True:
        user_input = input("你: ")
        
        if user_input.lower() == 'quit':
            print("机器人:", chatbot.chat("再见"))
            break
        elif user_input.startswith('personality:'):
            personality = user_input.split(':')[1]
            print("系统:", chatbot.set_personality(personality))
            continue
        elif user_input.lower() == 'summary':
            summary = chatbot.get_conversation_summary()
            print("对话摘要:", json.dumps(summary, ensure_ascii=False, indent=2))
            continue
        
        response = chatbot.chat(user_input)
        print("机器人:", response)


def demonstrate_chatbot_features():
    """演示聊天机器人功能"""
    print("=== 聊天机器人功能演示 ===")
    
    chatbot = Chatbot()
    
    # 测试不同类型的对话
    test_conversations = [
        "你好！",
        "今天天气怎么样？",
        "什么是人工智能？",
        "请推荐一些好吃的pizza",
        "我想预订明天8点的位置",
        "爱因斯坦是谁？",
        "我对你们的服务很不满意！",
        "再见"
    ]
    
    print("自动对话演示:")
    for user_msg in test_conversations:
        response = chatbot.chat(user_msg)
        print(f"用户: {user_msg}")
        print(f"机器人: {response}")
        print("-" * 30)
    
    # 展示情感分析
    print("\n情感分析演示:")
    sentiment_texts = [
        "我非常喜欢这个产品！",
        "这个服务太糟糕了！",
        "今天天气不错。"
    ]
    
    for text in sentiment_texts:
        sentiment = chatbot.sentiment_analyzer.analyze(text)
        print(f"文本: {text}")
        print(f"情感: {sentiment}")
        print("-" * 20)
    
    # 展示知识库查询
    print("\n知识库查询演示:")
    queries = ["爱因斯坦", "Python", "北京"]
    for query in queries:
        properties = chatbot.kb.query_entity(query)
        print(f"查询 '{query}': {properties}")


def main():
    """主演示函数"""
    print("高级AI应用：智能聊天机器人")
    print("整合了NLP、知识表示、对话管理、情感分析等技术")
    
    demonstrate_chatbot_features()
    
    print("\n选择运行模式:")
    print("1. 自动演示模式")
    print("2. 交互对话模式")
    
    choice = input("请选择 (1/2): ")
    
    if choice == "2":
        simulate_conversation()
    
    print("\n=== 聊天机器人技术总结 ===")
    print("1. 自然语言理解: 意图识别和实体提取")
    print("2. 对话管理: 状态跟踪和策略选择")
    print("3. 知识表示: 结构化知识存储和推理")
    print("4. 情感分析: 理解用户情感状态")
    print("5. 个性化: 不同的回复风格和个性")
    print("6. 上下文管理: 维护对话连贯性")


if __name__ == "__main__":
    main() 