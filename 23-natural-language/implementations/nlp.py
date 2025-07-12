"""
自然语言处理实现

包含词法分析、句法分析、语义分析、机器翻译等
"""

import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
import random
from abc import ABC, abstractmethod


class Tokenizer:
    """分词器"""
    
    def __init__(self):
        self.word_pattern = re.compile(r'\w+|[^\w\s]')
    
    def tokenize(self, text: str) -> List[str]:
        """分词"""
        return self.word_pattern.findall(text.lower())
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """句子分词"""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]


class NGramLanguageModel:
    """N-gram语言模型"""
    
    def __init__(self, n: int = 2):
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
    
    def train(self, texts: List[str]):
        """训练模型"""
        tokenizer = Tokenizer()
        
        for text in texts:
            tokens = tokenizer.tokenize(text)
            tokens = ['<start>'] * (self.n - 1) + tokens + ['<end>']
            
            # 更新词汇表
            self.vocabulary.update(tokens)
            
            # 统计n-gram
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = tuple(tokens[i:i + self.n - 1])
                
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1
    
    def get_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """获取条件概率"""
        ngram = context + (word,)
        
        if self.context_counts[context] == 0:
            return 1.0 / len(self.vocabulary)
        
        # 拉普拉斯平滑
        numerator = self.ngram_counts[ngram] + 1
        denominator = self.context_counts[context] + len(self.vocabulary)
        
        return numerator / denominator
    
    def generate_sentence(self, max_length: int = 20) -> str:
        """生成句子"""
        tokens = ['<start>'] * (self.n - 1)
        
        for _ in range(max_length):
            context = tuple(tokens[-(self.n - 1):])
            
            # 获取所有可能的下一个词
            candidates = []
            probabilities = []
            
            for word in self.vocabulary:
                if word not in ['<start>', '<end>']:
                    prob = self.get_probability(word, context)
                    candidates.append(word)
                    probabilities.append(prob)
            
            # 归一化概率
            total_prob = sum(probabilities)
            if total_prob > 0:
                probabilities = [p / total_prob for p in probabilities]
            
            # 选择下一个词
            if candidates:
                next_word = np.random.choice(candidates, p=probabilities)
                tokens.append(next_word)
                
                if next_word == '<end>':
                    break
        
        # 移除特殊标记
        result = [token for token in tokens if token not in ['<start>', '<end>']]
        return ' '.join(result)
    
    def perplexity(self, text: str) -> float:
        """计算困惑度"""
        tokenizer = Tokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = ['<start>'] * (self.n - 1) + tokens + ['<end>']
        
        log_prob = 0
        n_tokens = 0
        
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - self.n + 1:i])
            word = tokens[i]
            
            prob = self.get_probability(word, context)
            log_prob += math.log(prob)
            n_tokens += 1
        
        return math.exp(-log_prob / n_tokens)


class POSTagger:
    """词性标注器"""
    
    def __init__(self):
        self.word_tag_counts = defaultdict(lambda: defaultdict(int))
        self.tag_counts = defaultdict(int)
        self.tag_transitions = defaultdict(lambda: defaultdict(int))
        self.tags = set()
    
    def train(self, tagged_sentences: List[List[Tuple[str, str]]]):
        """训练标注器"""
        for sentence in tagged_sentences:
            prev_tag = '<start>'
            
            for word, tag in sentence:
                self.word_tag_counts[word][tag] += 1
                self.tag_counts[tag] += 1
                self.tag_transitions[prev_tag][tag] += 1
                self.tags.add(tag)
                prev_tag = tag
    
    def tag_sentence(self, words: List[str]) -> List[Tuple[str, str]]:
        """标注句子"""
        if not words:
            return []
        
        # 维特比算法
        n = len(words)
        dp = [{} for _ in range(n)]
        path = [{} for _ in range(n)]
        
        # 初始化
        for tag in self.tags:
            emission_prob = self.get_emission_probability(words[0], tag)
            transition_prob = self.get_transition_probability('<start>', tag)
            dp[0][tag] = emission_prob * transition_prob
            path[0][tag] = []
        
        # 动态规划
        for t in range(1, n):
            for tag in self.tags:
                best_prob = 0
                best_path = []
                
                for prev_tag in self.tags:
                    if prev_tag in dp[t-1]:
                        emission_prob = self.get_emission_probability(words[t], tag)
                        transition_prob = self.get_transition_probability(prev_tag, tag)
                        prob = dp[t-1][prev_tag] * emission_prob * transition_prob
                        
                        if prob > best_prob:
                            best_prob = prob
                            best_path = path[t-1][prev_tag] + [prev_tag]
                
                dp[t][tag] = best_prob
                path[t][tag] = best_path
        
        # 回溯找最优路径
        best_tag = max(dp[n-1].keys(), key=lambda tag: dp[n-1][tag])
        best_path = path[n-1][best_tag] + [best_tag]
        
        return list(zip(words, best_path))
    
    def get_emission_probability(self, word: str, tag: str) -> float:
        """获取发射概率"""
        if self.tag_counts[tag] == 0:
            return 1e-10
        
        return (self.word_tag_counts[word][tag] + 1) / (self.tag_counts[tag] + len(self.word_tag_counts))
    
    def get_transition_probability(self, prev_tag: str, tag: str) -> float:
        """获取转移概率"""
        prev_tag_count = sum(self.tag_transitions[prev_tag].values())
        if prev_tag_count == 0:
            return 1.0 / len(self.tags)
        
        return (self.tag_transitions[prev_tag][tag] + 1) / (prev_tag_count + len(self.tags))


class ChartParser:
    """图表解析器"""
    
    def __init__(self):
        self.grammar = {}
        self.lexicon = {}
    
    def add_rule(self, lhs: str, rhs: List[str]):
        """添加语法规则"""
        if lhs not in self.grammar:
            self.grammar[lhs] = []
        self.grammar[lhs].append(rhs)
    
    def add_lexical_rule(self, word: str, pos: str):
        """添加词汇规则"""
        if word not in self.lexicon:
            self.lexicon[word] = []
        self.lexicon[word].append(pos)
    
    def parse(self, words: List[str]) -> List[Dict]:
        """解析句子"""
        n = len(words)
        # CKY算法
        chart = [[set() for _ in range(n)] for _ in range(n)]
        
        # 填充对角线（词汇项）
        for i in range(n):
            word = words[i]
            if word in self.lexicon:
                for pos in self.lexicon[word]:
                    chart[i][i].add(pos)
        
        # 填充图表
        for length in range(2, n + 1):  # 跨度长度
            for i in range(n - length + 1):
                j = i + length - 1
                
                # 尝试所有可能的分割点
                for k in range(i, j):
                    left_symbols = chart[i][k]
                    right_symbols = chart[k + 1][j]
                    
                    # 检查所有语法规则
                    for lhs, rhs_list in self.grammar.items():
                        for rhs in rhs_list:
                            if (len(rhs) == 2 and 
                                rhs[0] in left_symbols and 
                                rhs[1] in right_symbols):
                                chart[i][j].add(lhs)
        
        # 返回解析结果
        parses = []
        for symbol in chart[0][n-1]:
            parses.append({
                'symbol': symbol,
                'span': (0, n-1),
                'words': words
            })
        
        return parses


class TFIDFVectorizer:
    """TF-IDF向量化器"""
    
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self.vocabulary = {}
        self.idf_values = {}
        self.feature_names = []
    
    def fit(self, documents: List[str]):
        """训练向量化器"""
        tokenizer = Tokenizer()
        
        # 统计词频
        doc_freq = defaultdict(int)
        all_words = set()
        
        for doc in documents:
            words = set(tokenizer.tokenize(doc))
            all_words.update(words)
            
            for word in words:
                doc_freq[word] += 1
        
        # 计算IDF值
        n_docs = len(documents)
        for word in all_words:
            self.idf_values[word] = math.log(n_docs / doc_freq[word])
        
        # 选择特征
        sorted_words = sorted(doc_freq.items(), key=lambda x: x[1], reverse=True)
        self.feature_names = [word for word, freq in sorted_words[:self.max_features]]
        
        # 构建词汇表
        self.vocabulary = {word: idx for idx, word in enumerate(self.feature_names)}
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """转换文档为向量"""
        tokenizer = Tokenizer()
        n_docs = len(documents)
        n_features = len(self.feature_names)
        
        X = np.zeros((n_docs, n_features))
        
        for doc_idx, doc in enumerate(documents):
            words = tokenizer.tokenize(doc)
            word_counts = Counter(words)
            doc_length = len(words)
            
            for word, count in word_counts.items():
                if word in self.vocabulary:
                    feature_idx = self.vocabulary[word]
                    tf = count / doc_length
                    idf = self.idf_values.get(word, 0)
                    X[doc_idx, feature_idx] = tf * idf
        
        return X


class NaiveBayesClassifier:
    """朴素贝叶斯分类器"""
    
    def __init__(self):
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = []
    
    def train(self, X: np.ndarray, y: List[str]):
        """训练分类器"""
        self.classes = list(set(y))
        n_samples, n_features = X.shape
        
        # 计算类别先验概率
        class_counts = Counter(y)
        for class_name in self.classes:
            self.class_priors[class_name] = class_counts[class_name] / n_samples
        
        # 计算特征概率
        for class_name in self.classes:
            class_mask = [label == class_name for label in y]
            class_features = X[class_mask]
            
            # 拉普拉斯平滑
            feature_sums = np.sum(class_features, axis=0) + 1
            total_sum = np.sum(feature_sums)
            
            self.feature_probs[class_name] = feature_sums / total_sum
    
    def predict(self, X: np.ndarray) -> List[str]:
        """预测"""
        predictions = []
        
        for sample in X:
            best_class = None
            best_prob = float('-inf')
            
            for class_name in self.classes:
                # 计算后验概率的对数
                log_prob = math.log(self.class_priors[class_name])
                
                for feature_idx, feature_value in enumerate(sample):
                    if feature_value > 0:
                        log_prob += math.log(self.feature_probs[class_name][feature_idx])
                
                if log_prob > best_prob:
                    best_prob = log_prob
                    best_class = class_name
            
            predictions.append(best_class)
        
        return predictions


class WordEmbedding:
    """词向量（简化版Word2Vec）"""
    
    def __init__(self, vector_size: int = 100, window_size: int = 5):
        self.vector_size = vector_size
        self.window_size = window_size
        self.word_to_index = {}
        self.index_to_word = {}
        self.embeddings = None
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """构建词汇表"""
        tokenizer = Tokenizer()
        word_counts = Counter()
        
        for text in texts:
            words = tokenizer.tokenize(text)
            word_counts.update(words)
        
        # 只保留频率较高的词
        vocab_words = [word for word, count in word_counts.most_common(5000)]
        
        self.word_to_index = {word: idx for idx, word in enumerate(vocab_words)}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}
        
        # 初始化词向量
        vocab_size = len(vocab_words)
        self.embeddings = np.random.normal(0, 0.1, (vocab_size, self.vector_size))
    
    def train(self, texts: List[str], epochs: int = 5, learning_rate: float = 0.01):
        """训练词向量"""
        tokenizer = Tokenizer()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for text in texts:
                words = tokenizer.tokenize(text)
                
                # 生成训练对
                for center_idx, center_word in enumerate(words):
                    if center_word not in self.word_to_index:
                        continue
                    
                    center_word_idx = self.word_to_index[center_word]
                    
                    # 上下文窗口
                    start = max(0, center_idx - self.window_size)
                    end = min(len(words), center_idx + self.window_size + 1)
                    
                    for context_idx in range(start, end):
                        if context_idx == center_idx:
                            continue
                        
                        context_word = words[context_idx]
                        if context_word not in self.word_to_index:
                            continue
                        
                        context_word_idx = self.word_to_index[context_word]
                        
                        # 简化的Skip-gram训练
                        loss = self.train_pair(center_word_idx, context_word_idx, learning_rate)
                        total_loss += loss
            
            print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
    
    def train_pair(self, center_idx: int, context_idx: int, learning_rate: float) -> float:
        """训练词对"""
        # 简化的负采样
        center_vector = self.embeddings[center_idx]
        context_vector = self.embeddings[context_idx]
        
        # 计算相似度
        similarity = np.dot(center_vector, context_vector)
        prob = 1 / (1 + np.exp(-similarity))
        
        # 计算损失
        loss = -np.log(prob + 1e-10)
        
        # 更新向量
        gradient = (prob - 1) * learning_rate
        self.embeddings[center_idx] += gradient * context_vector
        self.embeddings[context_idx] += gradient * center_vector
        
        return loss
    
    def most_similar(self, word: str, topn: int = 5) -> List[Tuple[str, float]]:
        """找到最相似的词"""
        if word not in self.word_to_index:
            return []
        
        word_idx = self.word_to_index[word]
        word_vector = self.embeddings[word_idx]
        
        # 计算余弦相似度
        similarities = []
        for idx, other_word in self.index_to_word.items():
            if idx == word_idx:
                continue
            
            other_vector = self.embeddings[idx]
            similarity = np.dot(word_vector, other_vector) / (
                np.linalg.norm(word_vector) * np.linalg.norm(other_vector)
            )
            similarities.append((other_word, similarity))
        
        # 排序并返回前topn个
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]


def demo_language_model():
    """演示语言模型"""
    print("N-gram语言模型演示")
    print("=" * 30)
    
    # 训练数据
    texts = [
        "The cat sat on the mat.",
        "The dog ran in the park.",
        "A cat and a dog played together.",
        "The sun is shining today.",
        "I love natural language processing."
    ]
    
    # 训练模型
    model = NGramLanguageModel(n=2)
    model.train(texts)
    
    # 生成句子
    for i in range(3):
        sentence = model.generate_sentence()
        print(f"生成的句子 {i+1}: {sentence}")
    
    # 计算困惑度
    test_text = "The cat ran in the park."
    perplexity = model.perplexity(test_text)
    print(f"\n测试文本困惑度: {perplexity:.2f}")


def demo_pos_tagging():
    """演示词性标注"""
    print("词性标注演示")
    print("=" * 30)
    
    # 训练数据
    tagged_sentences = [
        [("The", "DT"), ("cat", "NN"), ("sat", "VBD"), ("on", "IN"), ("the", "DT"), ("mat", "NN")],
        [("Dogs", "NNS"), ("run", "VBP"), ("fast", "RB")],
        [("I", "PRP"), ("love", "VBP"), ("programming", "VBG")],
        [("She", "PRP"), ("is", "VBZ"), ("reading", "VBG"), ("a", "DT"), ("book", "NN")]
    ]
    
    # 训练标注器
    tagger = POSTagger()
    tagger.train(tagged_sentences)
    
    # 标注新句子
    test_sentence = ["The", "dog", "is", "running"]
    tagged = tagger.tag_sentence(test_sentence)
    
    print("标注结果:")
    for word, tag in tagged:
        print(f"{word}: {tag}")


def demo_text_classification():
    """演示文本分类"""
    print("文本分类演示")
    print("=" * 30)
    
    # 训练数据
    documents = [
        "I love this movie. It's amazing!",
        "This film is terrible. I hate it.",
        "Great acting and wonderful story.",
        "Boring movie. Waste of time.",
        "Excellent cinematography and direction.",
        "Poor script and bad acting."
    ]
    
    labels = ["positive", "negative", "positive", "negative", "positive", "negative"]
    
    # 特征提取
    vectorizer = TFIDFVectorizer(max_features=50)
    vectorizer.fit(documents)
    X = vectorizer.transform(documents)
    
    # 训练分类器
    classifier = NaiveBayesClassifier()
    classifier.train(X, labels)
    
    # 测试
    test_docs = [
        "I really enjoyed this movie.",
        "This film is awful and boring."
    ]
    
    X_test = vectorizer.transform(test_docs)
    predictions = classifier.predict(X_test)
    
    print("分类结果:")
    for doc, pred in zip(test_docs, predictions):
        print(f"'{doc}' -> {pred}")


def demo_word_embeddings():
    """演示词向量"""
    print("词向量演示")
    print("=" * 30)
    
    # 训练数据
    texts = [
        "The cat sat on the mat",
        "The dog ran in the park",
        "A cat and a dog played together",
        "The sun is shining today",
        "I love natural language processing",
        "Machine learning is fascinating",
        "Deep learning uses neural networks",
        "Natural language understanding is challenging"
    ]
    
    # 训练词向量
    embedding = WordEmbedding(vector_size=50, window_size=2)
    embedding.build_vocabulary(texts)
    embedding.train(texts, epochs=10)
    
    # 寻找相似词
    test_words = ["cat", "dog", "language"]
    for word in test_words:
        similar = embedding.most_similar(word, topn=3)
        print(f"\n与 '{word}' 相似的词:")
        for sim_word, similarity in similar:
            print(f"  {sim_word}: {similarity:.3f}")


if __name__ == "__main__":
    # 演示不同的自然语言处理技术
    demo_language_model()
    print("\n" + "="*50)
    demo_pos_tagging()
    print("\n" + "="*50)
    demo_text_classification()
    print("\n" + "="*50)
    demo_word_embeddings()
    
    print("\n✅ 自然语言处理演示完成！") 