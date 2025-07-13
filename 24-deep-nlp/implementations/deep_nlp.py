#!/usr/bin/env python3
"""
第24章：深度自然语言处理 (Deep Natural Language Processing)

本模块实现了深度学习在自然语言处理中的应用：
- 词向量 (Word Embeddings)
- 序列到序列模型 (Seq2Seq)
- 注意力机制 (Attention Mechanism)
- Transformer 架构
- BERT 类模型基础
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import re
from collections import Counter, defaultdict
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class WordEmbedding:
    """词向量模型"""
    
    def __init__(self, embedding_dim: int = 100):
        self.embedding_dim = embedding_dim
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.embeddings = None
        self.vocab_size = 0
    
    def build_vocab(self, corpus: List[str], min_count: int = 1):
        """构建词汇表"""
        word_counts = Counter()
        
        for sentence in corpus:
            words = self._tokenize(sentence)
            word_counts.update(words)
        
        # 过滤低频词
        vocab = [word for word, count in word_counts.items() if count >= min_count]
        
        # 添加特殊标记
        vocab = ['<PAD>', '<UNK>', '<SOS>', '<EOS>'] + vocab
        
        self.word_to_idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(vocab)
        
        # 初始化随机词向量
        self.embeddings = np.random.normal(0, 0.1, (self.vocab_size, self.embedding_dim))
    
    def _tokenize(self, text: str) -> List[str]:
        """简单分词"""
        # 处理中文和英文
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text.lower())
        # 中文字符分割
        words = []
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # 中文字符
                words.append(char)
            elif char.isalnum():
                words.append(char)
        
        # 合并连续的英文字符
        result = []
        temp_word = ""
        for word in words:
            if word.isalpha() and not ('\u4e00' <= word <= '\u9fff'):
                temp_word += word
            else:
                if temp_word:
                    result.append(temp_word)
                    temp_word = ""
                if word and not word.isspace():
                    result.append(word)
        if temp_word:
            result.append(temp_word)
        
        return [w for w in result if w.strip()]
    
    def text_to_indices(self, text: str) -> List[int]:
        """文本转索引"""
        words = self._tokenize(text)
        return [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
    
    def indices_to_text(self, indices: List[int]) -> str:
        """索引转文本"""
        words = [self.idx_to_word.get(idx, '<UNK>') for idx in indices]
        return ' '.join(words)
    
    def get_embedding(self, word: str) -> np.ndarray:
        """获取词向量"""
        idx = self.word_to_idx.get(word, self.word_to_idx['<UNK>'])
        return self.embeddings[idx]
    
    def similarity(self, word1: str, word2: str) -> float:
        """计算词相似度"""
        emb1 = self.get_embedding(word1)
        emb2 = self.get_embedding(word2)
        
        # 余弦相似度
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return np.dot(emb1, emb2) / (norm1 * norm2)
    
    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """找到最相似的词"""
        if word not in self.word_to_idx:
            return []
        
        target_emb = self.get_embedding(word)
        similarities = []
        
        for w, idx in self.word_to_idx.items():
            if w != word and w not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
                sim = self.similarity(word, w)
                similarities.append((w, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

class SimpleRNN:
    """简单RNN实现"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 权重初始化
        self.Wxh = np.random.randn(hidden_dim, input_dim) * 0.01
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.Why = np.random.randn(output_dim, hidden_dim) * 0.01
        self.bh = np.zeros((hidden_dim, 1))
        self.by = np.zeros((output_dim, 1))
    
    def forward(self, inputs: List[np.ndarray], initial_hidden: np.ndarray = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """前向传播"""
        if initial_hidden is None:
            h = np.zeros((self.hidden_dim, 1))
        else:
            h = initial_hidden
        
        hidden_states = []
        outputs = []
        
        for x in inputs:
            x = x.reshape(-1, 1)
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            
            hidden_states.append(h.copy())
            outputs.append(y.copy())
        
        return outputs, hidden_states
    
    def generate(self, seed_idx: int, length: int, temperature: float = 1.0) -> List[int]:
        """生成序列"""
        h = np.zeros((self.hidden_dim, 1))
        generated = [seed_idx]
        
        for _ in range(length):
            x = np.zeros((self.input_dim, 1))
            x[generated[-1]] = 1
            
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            
            # 温度采样
            p = self._softmax(y.flatten() / temperature)
            next_idx = np.random.choice(len(p), p=p)
            generated.append(next_idx)
        
        return generated[1:]  # 排除种子
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class AttentionMechanism:
    """注意力机制"""
    
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim
        self.W_attention = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.v_attention = np.random.randn(hidden_dim, 1) * 0.01
    
    def compute_attention(self, query: np.ndarray, keys: List[np.ndarray], 
                         values: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """计算注意力权重和输出"""
        # 计算注意力分数
        scores = []
        for key in keys:
            # 简化的注意力计算
            combined = np.tanh(np.dot(self.W_attention, query) + 
                             np.dot(self.W_attention, key))
            score = np.dot(self.v_attention.T, combined)
            scores.append(score[0, 0])
        
        # Softmax归一化
        scores = np.array(scores)
        attention_weights = self._softmax(scores)
        
        # 加权求和
        context = np.zeros_like(values[0])
        for i, (weight, value) in enumerate(zip(attention_weights, values)):
            context += weight * value
        
        return context, attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax函数"""
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

class SimpleTransformer:
    """简化的Transformer实现"""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, n_layers: int = 6):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # 词嵌入和位置编码
        self.word_embeddings = np.random.randn(vocab_size, d_model) * 0.01
        self.position_embeddings = self._create_positional_encoding(1000, d_model)
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> np.ndarray:
        """创建位置编码"""
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def embed(self, tokens: List[int]) -> np.ndarray:
        """嵌入层"""
        seq_len = len(tokens)
        
        # 词嵌入
        embeddings = np.array([self.word_embeddings[token] for token in tokens])
        
        # 位置编码
        pos_encodings = self.position_embeddings[:seq_len]
        
        return embeddings + pos_encodings
    
    def self_attention(self, x: np.ndarray) -> np.ndarray:
        """简化的自注意力"""
        seq_len, d_model = x.shape
        
        # Q, K, V 矩阵 (简化实现)
        Q = K = V = x
        
        # 计算注意力分数
        scores = np.dot(Q, K.T) / np.sqrt(d_model)
        
        # Softmax
        attention_weights = self._softmax_2d(scores)
        
        # 加权求和
        output = np.dot(attention_weights, V)
        
        return output
    
    def _softmax_2d(self, x: np.ndarray) -> np.ndarray:
        """2D Softmax"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class BERTLike:
    """类BERT模型基础框架"""
    
    def __init__(self, vocab_size: int, hidden_size: int = 768):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embeddings = np.random.randn(vocab_size, hidden_size) * 0.01
        
        # 特殊标记
        self.CLS_TOKEN = 0
        self.SEP_TOKEN = 1
        self.MASK_TOKEN = 2
        self.PAD_TOKEN = 3
    
    def tokenize_and_mask(self, text: str, mask_prob: float = 0.15) -> Tuple[List[int], List[int]]:
        """分词并随机遮蔽"""
        # 简化的分词
        tokens = text.lower().split()
        token_ids = [hash(token) % (self.vocab_size - 10) + 10 for token in tokens]  # 简化映射
        
        # 添加特殊标记
        token_ids = [self.CLS_TOKEN] + token_ids + [self.SEP_TOKEN]
        
        # 随机遮蔽
        masked_ids = token_ids.copy()
        labels = [-1] * len(token_ids)  # -1表示不预测
        
        for i in range(1, len(token_ids) - 1):  # 不遮蔽CLS和SEP
            if np.random.random() < mask_prob:
                labels[i] = token_ids[i]  # 保存原始token作为标签
                
                rand = np.random.random()
                if rand < 0.8:
                    masked_ids[i] = self.MASK_TOKEN
                elif rand < 0.9:
                    masked_ids[i] = np.random.randint(4, self.vocab_size)
                # 10%概率保持原样
        
        return masked_ids, labels
    
    def predict_masked_tokens(self, masked_ids: List[int], labels: List[int]) -> Dict[str, Any]:
        """预测被遮蔽的词（简化实现）"""
        predictions = {}
        
        for i, (masked_id, label) in enumerate(zip(masked_ids, labels)):
            if label != -1:  # 需要预测的位置
                # 简化的预测逻辑
                if masked_id == self.MASK_TOKEN:
                    # 随机预测一个词
                    pred = np.random.randint(4, self.vocab_size)
                    confidence = 0.3 + np.random.random() * 0.4
                else:
                    pred = masked_id
                    confidence = 0.8 + np.random.random() * 0.2
                
                predictions[i] = {
                    'predicted': pred,
                    'actual': label,
                    'confidence': confidence,
                    'correct': pred == label
                }
        
        return predictions

class TextClassifier:
    """文本分类器"""
    
    def __init__(self, embedding_dim: int = 100, hidden_dim: int = 128):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.classes = []
        self.word_embeddings = None
        
        # 简化的神经网络层
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
    
    def build_model(self, num_classes: int, vocab_size: int):
        """构建模型"""
        self.classes = list(range(num_classes))
        
        # 初始化权重
        self.word_embeddings = np.random.randn(vocab_size, self.embedding_dim) * 0.01
        self.W1 = np.random.randn(self.hidden_dim, self.embedding_dim) * 0.01
        self.b1 = np.zeros((self.hidden_dim, 1))
        self.W2 = np.random.randn(num_classes, self.hidden_dim) * 0.01
        self.b2 = np.zeros((num_classes, 1))
    
    def encode_text(self, token_ids: List[int]) -> np.ndarray:
        """编码文本"""
        if not token_ids:
            return np.zeros((self.embedding_dim, 1))
        
        # 平均词向量
        embeddings = np.array([self.word_embeddings[idx] for idx in token_ids])
        mean_embedding = np.mean(embeddings, axis=0).reshape(-1, 1)
        
        return mean_embedding
    
    def predict(self, token_ids: List[int]) -> Tuple[int, List[float]]:
        """预测分类"""
        # 编码
        x = self.encode_text(token_ids)
        
        # 前向传播
        h = np.tanh(np.dot(self.W1, x) + self.b1)
        scores = np.dot(self.W2, h) + self.b2
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / np.sum(exp_scores)
        
        predicted_class = np.argmax(probs)
        
        return predicted_class, probs.flatten().tolist()

def demo_word_embeddings():
    """演示词向量"""
    print("\n" + "="*50)
    print("词向量演示")
    print("="*50)
    
    # 创建简单语料库
    corpus = [
        "我喜欢吃苹果",
        "苹果很甜很好吃",
        "我不喜欢吃橘子",
        "橘子太酸了",
        "香蕉很甜",
        "我喜欢香蕉",
        "水果都很好吃",
        "苹果和香蕉都是水果"
    ]
    
    print("\n训练语料:")
    for i, sentence in enumerate(corpus, 1):
        print(f"  {i}. {sentence}")
    
    # 训练词向量
    we = WordEmbedding(embedding_dim=50)
    we.build_vocab(corpus)
    
    print(f"\n词汇表大小: {we.vocab_size}")
    print(f"词汇表前20个词: {list(we.word_to_idx.keys())[:20]}")
    
    # 计算词相似度
    print(f"\n词相似度:")
    word_pairs = [("苹果", "香蕉"), ("喜欢", "好吃"), ("甜", "酸"), ("水果", "苹果")]
    
    for word1, word2 in word_pairs:
        if word1 in we.word_to_idx and word2 in we.word_to_idx:
            sim = we.similarity(word1, word2)
            print(f"  {word1} - {word2}: {sim:.3f}")
    
    # 寻找最相似词
    test_word = "苹果"
    if test_word in we.word_to_idx:
        similar_words = we.most_similar(test_word, top_k=3)
        print(f"\n与'{test_word}'最相似的词:")
        for word, sim in similar_words:
            print(f"  {word}: {sim:.3f}")

def demo_rnn_text_generation():
    """演示RNN文本生成"""
    print("\n" + "="*50)
    print("RNN文本生成演示")
    print("="*50)
    
    # 简单字符级RNN
    text = "hello world this is a simple text for rnn training"
    chars = list(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    
    print(f"字符集: {chars}")
    print(f"字符集大小: {len(chars)}")
    
    # 创建RNN
    rnn = SimpleRNN(input_dim=len(chars), hidden_dim=20, output_dim=len(chars))
    
    # 模拟生成
    print(f"\n生成文本示例 (随机初始化权重):")
    seed_char = 'h'
    if seed_char in char_to_idx:
        seed_idx = char_to_idx[seed_char]
        generated_indices = rnn.generate(seed_idx, length=10, temperature=1.0)
        generated_text = ''.join([idx_to_char.get(idx, '?') for idx in generated_indices])
        print(f"  种子字符: '{seed_char}'")
        print(f"  生成序列: '{generated_text}'")
        print("  (注: 权重未训练，生成结果为随机)")

def demo_attention_mechanism():
    """演示注意力机制"""
    print("\n" + "="*50)
    print("注意力机制演示")
    print("="*50)
    
    # 创建注意力机制
    attention = AttentionMechanism(hidden_dim=10)
    
    # 模拟编码器隐状态
    encoder_states = [
        np.random.randn(10, 1),
        np.random.randn(10, 1),
        np.random.randn(10, 1),
        np.random.randn(10, 1)
    ]
    
    # 模拟解码器查询
    decoder_query = np.random.randn(10, 1)
    
    # 计算注意力
    context, weights = attention.compute_attention(decoder_query, encoder_states, encoder_states)
    
    print(f"编码器状态数量: {len(encoder_states)}")
    print(f"注意力权重: {weights}")
    print(f"权重和: {np.sum(weights):.3f}")
    print(f"上下文向量形状: {context.shape}")
    
    # 可视化注意力权重
    positions = list(range(len(weights)))
    plt.figure(figsize=(8, 4))
    plt.bar(positions, weights, alpha=0.7)
    plt.xlabel('编码器位置')
    plt.ylabel('注意力权重')
    plt.title('注意力权重分布')
    plt.xticks(positions)
    for i, w in enumerate(weights):
        plt.text(i, w + 0.01, f'{w:.2f}', ha='center')
    plt.tight_layout()
    plt.savefig('attention_weights.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("注意力权重图已保存为 'attention_weights.png'")

def demo_transformer():
    """演示Transformer"""
    print("\n" + "="*50)
    print("Transformer演示")
    print("="*50)
    
    # 创建简化的Transformer
    vocab_size = 1000
    transformer = SimpleTransformer(vocab_size=vocab_size, d_model=128, n_heads=4, n_layers=2)
    
    # 示例输入
    input_tokens = [10, 25, 67, 89, 156]  # 模拟token IDs
    
    print(f"输入token: {input_tokens}")
    print(f"序列长度: {len(input_tokens)}")
    
    # 嵌入
    embeddings = transformer.embed(input_tokens)
    print(f"嵌入形状: {embeddings.shape}")
    
    # 自注意力
    attention_output = transformer.self_attention(embeddings)
    print(f"自注意力输出形状: {attention_output.shape}")
    
    # 位置编码可视化
    pos_encoding = transformer.position_embeddings[:20, :10]  # 前20个位置，前10个维度
    
    plt.figure(figsize=(10, 6))
    plt.imshow(pos_encoding.T, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.xlabel('位置')
    plt.ylabel('维度')
    plt.title('位置编码可视化')
    plt.tight_layout()
    plt.savefig('positional_encoding.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("位置编码图已保存为 'positional_encoding.png'")

def demo_bert_like():
    """演示BERT类模型"""
    print("\n" + "="*50)
    print("BERT类模型演示")
    print("="*50)
    
    # 创建BERT类模型
    bert = BERTLike(vocab_size=1000, hidden_size=256)
    
    # 示例文本
    text = "人工智能是计算机科学的一个分支"
    print(f"原始文本: {text}")
    
    # 分词和遮蔽
    masked_ids, labels = bert.tokenize_and_mask(text, mask_prob=0.3)
    
    print(f"\n分词结果:")
    print(f"Token IDs: {masked_ids}")
    print(f"Labels: {labels}")
    
    # 显示遮蔽位置
    masked_positions = [i for i, label in enumerate(labels) if label != -1]
    print(f"\n遮蔽位置: {masked_positions}")
    
    # 预测遮蔽词
    predictions = bert.predict_masked_tokens(masked_ids, labels)
    
    print(f"\n遮蔽语言模型预测:")
    for pos, pred_info in predictions.items():
        status = "✓" if pred_info['correct'] else "✗"
        print(f"  位置 {pos}: 预测={pred_info['predicted']}, "
              f"实际={pred_info['actual']}, "
              f"置信度={pred_info['confidence']:.2f} {status}")

def demo_text_classification():
    """演示文本分类"""
    print("\n" + "="*50)
    print("文本分类演示")
    print("="*50)
    
    # 创建分类器
    classifier = TextClassifier(embedding_dim=50, hidden_dim=32)
    classifier.build_model(num_classes=3, vocab_size=100)
    
    # 模拟数据
    texts = [
        "这部电影真的很好看",
        "电影情节很无聊",
        "演员表演很一般",
        "特效制作很精彩",
        "剧情发展很缓慢"
    ]
    
    labels = ["正面", "负面", "中性"]
    
    print(f"分类类别: {labels}")
    print(f"\n文本分类结果:")
    
    for i, text in enumerate(texts):
        # 简化的token化
        token_ids = [hash(char) % 90 + 10 for char in text if char.strip()]
        
        predicted_class, probs = classifier.predict(token_ids)
        predicted_label = labels[predicted_class]
        confidence = max(probs)
        
        print(f"  文本: '{text}'")
        print(f"  预测: {predicted_label} (置信度: {confidence:.2f})")
        print(f"  概率分布: {[f'{label}:{prob:.2f}' for label, prob in zip(labels, probs)]}")
        print()

def visualize_embedding_space():
    """可视化词向量空间"""
    print("\n" + "="*50)
    print("词向量空间可视化")
    print("="*50)
    
    # 创建一些示例词向量
    words = ['国王', '女王', '男人', '女人', '苹果', '橘子', '汽车', '飞机']
    embeddings = np.random.randn(len(words), 2)  # 2D用于可视化
    
    # 手动调整一些向量以显示关系
    embeddings[0] = [1, 1]    # 国王
    embeddings[1] = [1, -1]   # 女王  
    embeddings[2] = [0.8, 0.9]  # 男人
    embeddings[3] = [0.8, -0.9] # 女人
    embeddings[4] = [-1, 0]   # 苹果
    embeddings[5] = [-0.8, 0.2] # 橘子
    embeddings[6] = [0, 1.2]   # 汽车
    embeddings[7] = [0.2, 1.5] # 飞机
    
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings[:, 0], embeddings[:, 1], s=100, alpha=0.7)
    
    for i, word in enumerate(words):
        plt.annotate(word, (embeddings[i, 0], embeddings[i, 1]), 
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=12, ha='left')
    
    # 绘制一些关系向量
    # 性别关系: 国王->女王, 男人->女人
    plt.arrow(embeddings[0, 0], embeddings[0, 1], 
             embeddings[1, 0] - embeddings[0, 0], embeddings[1, 1] - embeddings[0, 1],
             head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6)
    plt.arrow(embeddings[2, 0], embeddings[2, 1], 
             embeddings[3, 0] - embeddings[2, 0], embeddings[3, 1] - embeddings[2, 1],
             head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.6)
    
    plt.xlabel('维度 1')
    plt.ylabel('维度 2')
    plt.title('词向量空间可视化\n(红色箭头表示性别关系)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('word_embedding_space.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("词向量空间图已保存为 'word_embedding_space.png'")

def run_comprehensive_demo():
    """运行完整演示"""
    print("🤖 第24章：深度自然语言处理 - 完整演示")
    print("="*60)
    
    # 运行各个演示
    demo_word_embeddings()
    demo_rnn_text_generation()
    demo_attention_mechanism()
    demo_transformer()
    demo_bert_like()
    demo_text_classification()
    visualize_embedding_space()
    
    print("\n" + "="*60)
    print("深度自然语言处理演示完成！")
    print("="*60)
    print("\n📚 学习要点:")
    print("• 词向量将词汇映射到连续向量空间")
    print("• RNN可以处理序列数据但存在长期依赖问题")
    print("• 注意力机制解决了信息瓶颈问题")
    print("• Transformer通过自注意力实现并行化")
    print("• BERT等预训练模型开创了NLP新范式")
    print("• 深度学习极大提升了NLP任务性能")

if __name__ == "__main__":
    run_comprehensive_demo() 