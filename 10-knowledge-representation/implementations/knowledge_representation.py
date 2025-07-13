#!/usr/bin/env python3
"""
第10章：知识表示 (Knowledge Representation)

本模块实现了知识表示的核心概念：
- 本体工程
- 语义网络
- 描述逻辑
- 知识图谱
- 框架表示
"""

import json
from typing import Dict, List, Set, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

class RelationType(Enum):
    """关系类型"""
    IS_A = "is-a"
    PART_OF = "part-of"
    HAS_PROPERTY = "has-property"
    INSTANCE_OF = "instance-of"
    SIMILAR_TO = "similar-to"
    CAUSES = "causes"
    LOCATED_IN = "located-in"

@dataclass
class Concept:
    """概念"""
    name: str
    properties: Dict[str, Any] = field(default_factory=dict)
    parent_concepts: Set[str] = field(default_factory=set)
    child_concepts: Set[str] = field(default_factory=set)
    instances: Set[str] = field(default_factory=set)
    
    def add_property(self, property_name: str, value: Any):
        """添加属性"""
        self.properties[property_name] = value
    
    def inherit_properties(self, parent_concept: 'Concept'):
        """继承父概念的属性"""
        for prop, value in parent_concept.properties.items():
            if prop not in self.properties:
                self.properties[prop] = value

@dataclass
class Relation:
    """关系"""
    subject: str
    predicate: RelationType
    object: str
    confidence: float = 1.0
    
    def __str__(self):
        return f"{self.subject} --{self.predicate.value}--> {self.object}"

@dataclass
class Frame:
    """框架表示"""
    name: str
    slots: Dict[str, Any] = field(default_factory=dict)
    default_values: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def add_slot(self, slot_name: str, value: Any = None, default: Any = None, constraint: Any = None):
        """添加槽"""
        if value is not None:
            self.slots[slot_name] = value
        if default is not None:
            self.default_values[slot_name] = default
        if constraint is not None:
            self.constraints[slot_name] = constraint
    
    def get_value(self, slot_name: str) -> Any:
        """获取槽值"""
        if slot_name in self.slots:
            return self.slots[slot_name]
        elif slot_name in self.default_values:
            return self.default_values[slot_name]
        else:
            return None

class SemanticNetwork:
    """语义网络"""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.relations: List[Relation] = []
        self.relation_index: Dict[str, List[Relation]] = defaultdict(list)
    
    def add_concept(self, name: str, properties: Dict[str, Any] = None) -> Concept:
        """添加概念"""
        if name not in self.concepts:
            self.concepts[name] = Concept(name, properties or {})
        return self.concepts[name]
    
    def add_relation(self, subject: str, predicate: RelationType, object: str, confidence: float = 1.0):
        """添加关系"""
        # 确保概念存在
        self.add_concept(subject)
        self.add_concept(object)
        
        relation = Relation(subject, predicate, object, confidence)
        self.relations.append(relation)
        
        # 建立索引
        self.relation_index[subject].append(relation)
        
        # 更新概念间的关系
        if predicate == RelationType.IS_A:
            self.concepts[subject].parent_concepts.add(object)
            self.concepts[object].child_concepts.add(subject)
            # 继承属性
            self.concepts[subject].inherit_properties(self.concepts[object])
        elif predicate == RelationType.INSTANCE_OF:
            self.concepts[object].instances.add(subject)
    
    def get_relations(self, subject: str = None, predicate: RelationType = None, object: str = None) -> List[Relation]:
        """查询关系"""
        results = []
        
        for relation in self.relations:
            match = True
            if subject and relation.subject != subject:
                match = False
            if predicate and relation.predicate != predicate:
                match = False
            if object and relation.object != object:
                match = False
            
            if match:
                results.append(relation)
        
        return results
    
    def get_ancestors(self, concept_name: str) -> Set[str]:
        """获取概念的所有祖先"""
        ancestors = set()
        visited = set()
        queue = deque([concept_name])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            if current in self.concepts:
                for parent in self.concepts[current].parent_concepts:
                    if parent not in ancestors:
                        ancestors.add(parent)
                        queue.append(parent)
        
        return ancestors
    
    def get_descendants(self, concept_name: str) -> Set[str]:
        """获取概念的所有后代"""
        descendants = set()
        visited = set()
        queue = deque([concept_name])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            if current in self.concepts:
                for child in self.concepts[current].child_concepts:
                    if child not in descendants:
                        descendants.add(child)
                        queue.append(child)
        
        return descendants
    
    def is_subclass_of(self, subclass: str, superclass: str) -> bool:
        """检查是否为子类关系"""
        return superclass in self.get_ancestors(subclass)
    
    def find_common_ancestor(self, concept1: str, concept2: str) -> Optional[str]:
        """找到最近公共祖先"""
        ancestors1 = self.get_ancestors(concept1)
        ancestors2 = self.get_ancestors(concept2)
        
        common_ancestors = ancestors1.intersection(ancestors2)
        
        if not common_ancestors:
            return None
        
        # 找到最具体的公共祖先（启发式：最短路径）
        best_ancestor = None
        min_distance = float('inf')
        
        for ancestor in common_ancestors:
            dist1 = self._calculate_distance(concept1, ancestor)
            dist2 = self._calculate_distance(concept2, ancestor)
            total_dist = dist1 + dist2
            
            if total_dist < min_distance:
                min_distance = total_dist
                best_ancestor = ancestor
        
        return best_ancestor
    
    def _calculate_distance(self, start: str, end: str) -> int:
        """计算概念间的距离"""
        if start == end:
            return 0
        
        visited = set()
        queue = deque([(start, 0)])
        
        while queue:
            current, distance = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            if current == end:
                return distance
            
            if current in self.concepts:
                for parent in self.concepts[current].parent_concepts:
                    if parent not in visited:
                        queue.append((parent, distance + 1))
        
        return float('inf')

class Ontology:
    """本体"""
    
    def __init__(self, name: str):
        self.name = name
        self.classes: Dict[str, Dict[str, Any]] = {}
        self.properties: Dict[str, Dict[str, Any]] = {}
        self.individuals: Dict[str, Dict[str, Any]] = {}
        self.axioms: List[str] = []
    
    def add_class(self, class_name: str, superclasses: List[str] = None, properties: Dict[str, Any] = None):
        """添加类"""
        self.classes[class_name] = {
            'superclasses': superclasses or [],
            'properties': properties or {},
            'subclasses': [],
            'instances': []
        }
        
        # 更新父类的子类列表
        for superclass in (superclasses or []):
            if superclass in self.classes:
                self.classes[superclass]['subclasses'].append(class_name)
    
    def add_property(self, property_name: str, domain: str = None, range: str = None, 
                    property_type: str = "object"):
        """添加属性"""
        self.properties[property_name] = {
            'domain': domain,
            'range': range,
            'type': property_type  # object, data, annotation
        }
    
    def add_individual(self, individual_name: str, class_name: str, properties: Dict[str, Any] = None):
        """添加个体"""
        self.individuals[individual_name] = {
            'class': class_name,
            'properties': properties or {}
        }
        
        # 更新类的实例列表
        if class_name in self.classes:
            self.classes[class_name]['instances'].append(individual_name)
    
    def add_axiom(self, axiom: str):
        """添加公理"""
        self.axioms.append(axiom)
    
    def get_class_hierarchy(self) -> Dict[str, List[str]]:
        """获取类层次结构"""
        hierarchy = {}
        for class_name, class_info in self.classes.items():
            hierarchy[class_name] = class_info['subclasses']
        return hierarchy
    
    def infer_properties(self, individual_name: str) -> Dict[str, Any]:
        """推断个体的属性"""
        if individual_name not in self.individuals:
            return {}
        
        individual = self.individuals[individual_name]
        inferred_properties = individual['properties'].copy()
        
        # 从类继承属性
        class_name = individual['class']
        if class_name in self.classes:
            class_properties = self.classes[class_name]['properties']
            for prop, value in class_properties.items():
                if prop not in inferred_properties:
                    inferred_properties[prop] = value
        
        return inferred_properties
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'classes': self.classes,
            'properties': self.properties,
            'individuals': self.individuals,
            'axioms': self.axioms
        }

class FrameSystem:
    """框架系统"""
    
    def __init__(self):
        self.frames: Dict[str, Frame] = {}
        self.inheritance_hierarchy: Dict[str, List[str]] = defaultdict(list)
    
    def create_frame(self, name: str) -> Frame:
        """创建框架"""
        frame = Frame(name)
        self.frames[name] = frame
        return frame
    
    def add_inheritance(self, child_frame: str, parent_frame: str):
        """添加继承关系"""
        self.inheritance_hierarchy[child_frame].append(parent_frame)
    
    def get_inherited_value(self, frame_name: str, slot_name: str) -> Any:
        """获取继承的槽值"""
        if frame_name not in self.frames:
            return None
        
        frame = self.frames[frame_name]
        
        # 先检查自身
        value = frame.get_value(slot_name)
        if value is not None:
            return value
        
        # 递归检查父框架
        for parent_frame in self.inheritance_hierarchy[frame_name]:
            inherited_value = self.get_inherited_value(parent_frame, slot_name)
            if inherited_value is not None:
                return inherited_value
        
        return None

class KnowledgeGraph:
    """知识图谱"""
    
    def __init__(self):
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relations: List[Tuple[str, str, str]] = []
        self.entity_types: Dict[str, str] = {}
    
    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any] = None):
        """添加实体"""
        self.entities[entity_id] = properties or {}
        self.entity_types[entity_id] = entity_type
    
    def add_relation(self, subject: str, predicate: str, object: str):
        """添加关系三元组"""
        self.relations.append((subject, predicate, object))
    
    def query_by_entity(self, entity_id: str) -> List[Tuple[str, str, str]]:
        """查询实体相关的所有关系"""
        results = []
        for s, p, o in self.relations:
            if s == entity_id or o == entity_id:
                results.append((s, p, o))
        return results
    
    def query_by_relation(self, predicate: str) -> List[Tuple[str, str, str]]:
        """查询特定关系的所有三元组"""
        return [(s, p, o) for s, p, o in self.relations if p == predicate]
    
    def get_neighbors(self, entity_id: str) -> Set[str]:
        """获取实体的邻居"""
        neighbors = set()
        for s, p, o in self.relations:
            if s == entity_id:
                neighbors.add(o)
            elif o == entity_id:
                neighbors.add(s)
        return neighbors

def demo_semantic_network():
    """演示语义网络"""
    print("\n" + "="*50)
    print("语义网络演示")
    print("="*50)
    
    sn = SemanticNetwork()
    
    # 添加概念和关系
    print("\n构建动物分类语义网络:")
    
    # 动物层次
    sn.add_relation("Dog", RelationType.IS_A, "Mammal")
    sn.add_relation("Cat", RelationType.IS_A, "Mammal")
    sn.add_relation("Mammal", RelationType.IS_A, "Animal")
    sn.add_relation("Bird", RelationType.IS_A, "Animal")
    sn.add_relation("Robin", RelationType.IS_A, "Bird")
    
    # 添加属性
    sn.concepts["Animal"].add_property("can_move", True)
    sn.concepts["Animal"].add_property("has_life", True)
    sn.concepts["Mammal"].add_property("has_fur", True)
    sn.concepts["Mammal"].add_property("warm_blooded", True)
    sn.concepts["Bird"].add_property("can_fly", True)
    sn.concepts["Bird"].add_property("has_feathers", True)
    
    # 实例
    sn.add_relation("Fido", RelationType.INSTANCE_OF, "Dog")
    sn.add_relation("Tweety", RelationType.INSTANCE_OF, "Robin")
    
    print("概念层次:")
    for concept_name in ["Animal", "Mammal", "Dog", "Bird", "Robin"]:
        if concept_name in sn.concepts:
            concept = sn.concepts[concept_name]
            print(f"  {concept_name}: {concept.properties}")
    
    print(f"\nDog的祖先: {sn.get_ancestors('Dog')}")
    print(f"Animal的后代: {sn.get_descendants('Animal')}")
    print(f"Dog是否为Animal的子类: {sn.is_subclass_of('Dog', 'Animal')}")
    print(f"Dog和Robin的共同祖先: {sn.find_common_ancestor('Dog', 'Robin')}")

def demo_ontology():
    """演示本体"""
    print("\n" + "="*50)
    print("本体演示")
    print("="*50)
    
    # 创建大学本体
    university_ontology = Ontology("University")
    
    print("\n构建大学本体:")
    
    # 添加类
    university_ontology.add_class("Person")
    university_ontology.add_class("Student", ["Person"])
    university_ontology.add_class("Teacher", ["Person"])
    university_ontology.add_class("Course")
    university_ontology.add_class("Department")
    
    # 添加属性
    university_ontology.add_property("enrollsIn", "Student", "Course")
    university_ontology.add_property("teaches", "Teacher", "Course")
    university_ontology.add_property("worksIn", "Teacher", "Department")
    university_ontology.add_property("age", "Person", "Integer", "data")
    university_ontology.add_property("name", "Person", "String", "data")
    
    # 添加个体
    university_ontology.add_individual("John", "Student", {"age": 20, "name": "John Smith"})
    university_ontology.add_individual("Prof_Wang", "Teacher", {"age": 45, "name": "Wang Li"})
    university_ontology.add_individual("CS101", "Course", {"name": "Introduction to Computer Science"})
    university_ontology.add_individual("CS_Dept", "Department", {"name": "Computer Science Department"})
    
    # 添加公理
    university_ontology.add_axiom("Student ⊆ Person")
    university_ontology.add_axiom("Teacher ⊆ Person")
    university_ontology.add_axiom("∀x (Student(x) → ∃y (Course(y) ∧ enrollsIn(x,y)))")
    
    print("本体结构:")
    print(f"  类: {list(university_ontology.classes.keys())}")
    print(f"  属性: {list(university_ontology.properties.keys())}")
    print(f"  个体: {list(university_ontology.individuals.keys())}")
    print(f"  公理数量: {len(university_ontology.axioms)}")
    
    # 推断属性
    john_properties = university_ontology.infer_properties("John")
    print(f"\nJohn的推断属性: {john_properties}")

def demo_frame_system():
    """演示框架系统"""
    print("\n" + "="*50)
    print("框架系统演示")
    print("="*50)
    
    fs = FrameSystem()
    
    print("\n构建房间框架系统:")
    
    # 创建通用房间框架
    room_frame = fs.create_frame("Room")
    room_frame.add_slot("has_walls", True)
    room_frame.add_slot("has_floor", True)
    room_frame.add_slot("has_ceiling", True)
    room_frame.add_slot("temperature", default=20)
    room_frame.add_slot("lighting", default="normal")
    
    # 创建办公室框架
    office_frame = fs.create_frame("Office")
    office_frame.add_slot("has_desk", True)
    office_frame.add_slot("has_computer", True)
    office_frame.add_slot("working_hours", "9-17")
    fs.add_inheritance("Office", "Room")
    
    # 创建卧室框架
    bedroom_frame = fs.create_frame("Bedroom")
    bedroom_frame.add_slot("has_bed", True)
    bedroom_frame.add_slot("lighting", "dim")  # 覆盖默认值
    fs.add_inheritance("Bedroom", "Room")
    
    print("框架结构:")
    print(f"  Room框架: {room_frame.slots}")
    print(f"  Office框架: {office_frame.slots}")
    print(f"  Bedroom框架: {bedroom_frame.slots}")
    
    print("\n属性继承:")
    print(f"  Office的has_walls: {fs.get_inherited_value('Office', 'has_walls')}")
    print(f"  Office的temperature: {fs.get_inherited_value('Office', 'temperature')}")
    print(f"  Bedroom的lighting: {fs.get_inherited_value('Bedroom', 'lighting')}")

def demo_knowledge_graph():
    """演示知识图谱"""
    print("\n" + "="*50)
    print("知识图谱演示")
    print("="*50)
    
    kg = KnowledgeGraph()
    
    print("\n构建明星知识图谱:")
    
    # 添加实体
    kg.add_entity("Tom_Cruise", "Person", {"birth_year": 1962, "profession": "Actor"})
    kg.add_entity("Mission_Impossible", "Movie", {"year": 1996, "genre": "Action"})
    kg.add_entity("Top_Gun", "Movie", {"year": 1986, "genre": "Action"})
    kg.add_entity("Nicole_Kidman", "Person", {"birth_year": 1967, "profession": "Actress"})
    
    # 添加关系
    kg.add_relation("Tom_Cruise", "starred_in", "Mission_Impossible")
    kg.add_relation("Tom_Cruise", "starred_in", "Top_Gun")
    kg.add_relation("Tom_Cruise", "married_to", "Nicole_Kidman")
    kg.add_relation("Nicole_Kidman", "starred_in", "Top_Gun")
    
    print("实体:")
    for entity_id, properties in kg.entities.items():
        print(f"  {entity_id}: {properties}")
    
    print(f"\n关系三元组: {len(kg.relations)}个")
    for s, p, o in kg.relations:
        print(f"  {s} --{p}--> {o}")
    
    print(f"\nTom_Cruise的邻居: {kg.get_neighbors('Tom_Cruise')}")
    print(f"starred_in关系: {kg.query_by_relation('starred_in')}")

def run_comprehensive_demo():
    """运行完整演示"""
    print("🧠 第10章：知识表示 - 完整演示")
    print("="*60)
    
    # 运行各个演示
    demo_semantic_network()
    demo_ontology()
    demo_frame_system()
    demo_knowledge_graph()
    
    print("\n" + "="*60)
    print("知识表示演示完成！")
    print("="*60)
    print("\n📚 学习要点:")
    print("• 语义网络通过节点和边表示概念和关系")
    print("• 本体提供了领域知识的形式化表示")
    print("• 框架系统支持属性继承和默认值")
    print("• 知识图谱是大规模结构化知识的现代表示方法")
    print("• 不同的知识表示方法适用于不同的应用场景")

if __name__ == "__main__":
    run_comprehensive_demo() 