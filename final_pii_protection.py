"""
最终版本的PII保护系统
使用位置映射确保精确的替换，避免混乱
"""

import re
import json
import hashlib
import uuid
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from crypto_utils import UserDataEncryption, FieldEncryption


@dataclass
class PIIEntity:
    """PII实体类"""
    type: str
    value: str
    masked_value: str
    encrypted_value: str
    position: Tuple[int, int]
    priority: int = 0


class FinalPIIDetector:
    """最终版本的PII检测器"""
    
    def __init__(self):
        self.patterns = {
            'id_card': (r'(?<![0-9])\d{17}[\dXx](?![0-9])', 10),
            'bank_account': (r'(?<![0-9])\d{16,19}(?![0-9])(?![\dXx])', 9),
            'credit_card': (r'\b(?:\d{4}[-\s]?){3}\d{4}\b', 8),
            'ssn': (r'\b\d{3}-?\d{2}-?\d{4}\b', 7),
            'phone': (r'(?<![0-9])(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})(?![0-9])|(?<![0-9])1[3-9]\d{9}(?![0-9])|(?<![0-9])\+?86[-.\s]?1[3-9]\d{9}(?![0-9])|(?<![0-9])1[3-9]\d{1}[-.\s]?\d{4}[-.\s]?\d{4}(?![0-9])|(?<![0-9])\+?852[-.\s]?\d{8}(?![0-9])|(?<![0-9])852[-.\s]?\d{8}(?![0-9])|(?<![0-9])\d{8}(?![0-9])(?=.*香港|.*HK|.*Hong Kong)|(?<![0-9])\+?852[-.\s]?\d{4}[-.\s]?\d{4}(?![0-9])', 6),
            'email': (r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}', 5),
            'chinese_name': (r'(?<![，。！？\s])(?:我[是叫]|姓名|名字)[\u4e00-\u9fff]{2,4}|[\u4e00-\u9fff]{2,4}(?=[，。！？\s]|$)', 3),
            'english_name': (r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b(?![a-z])(?![A-Z])', 4)  # 英文姓名，后面不能跟小写字母或大写字母
        }
    
    def detect_pii(self, text: str) -> List[PIIEntity]:
        """检测文本中的PII"""
        all_entities = []
        
        for pii_type, (pattern, priority) in self.patterns.items():
            # 只有邮箱使用不区分大小写匹配
            flags = re.IGNORECASE if pii_type == 'email' else 0
            matches = re.finditer(pattern, text, flags)
            for match in matches:
                value = match.group()
                
                # 对中文姓名进行额外验证
                if pii_type == 'chinese_name' and not self._is_likely_name(value, text, match.start(), match.end()):
                    continue
                
                # 对英文姓名进行额外验证
                if pii_type == 'english_name' and not self._is_likely_english_name(value, text, match.start(), match.end()):
                    continue
                
                masked_value = self._mask_value(pii_type, value)
                
                entity = PIIEntity(
                    type=pii_type,
                    value=value,
                    masked_value=masked_value,
                    encrypted_value="",
                    position=(match.start(), match.end()),
                    priority=priority
                )
                all_entities.append(entity)
        
        # 解决重叠问题
        return self._resolve_overlaps(all_entities)
    
    def _is_likely_name(self, value: str, text: str, start: int, end: int) -> bool:
        """判断中文字符串是否可能是姓名"""
        # 排除常见的非姓名词汇
        non_name_words = {
            '科技股', '投资机会', '市场分析', '投资价值', '投资策略', '退休计划',
            '公司', '银行', '股票', '基金', '债券', '期货', '期权', '保险',
            '理财', '财务', '会计', '审计', '税务', '法律', '咨询', '服务',
            '产品', '技术', '研发', '销售', '市场', '运营', '管理', '行政',
            '人事', '培训', '教育', '医疗', '健康', '环境', '能源', '交通',
            '通信', '互联网', '软件', '硬件', '数据', '信息', '安全', '质量'
        }
        
        if value in non_name_words:
            return False
        
        # 检查上下文，看是否在姓名相关的语境中
        context_start = max(0, start - 10)
        context_end = min(len(text), end + 10)
        context = text[context_start:context_end]
        
        # 姓名相关的关键词
        name_indicators = ['我叫', '我是', '姓名', '名字', '先生', '女士', '小姐', '先生', '老师', '经理', '总', '主任']
        
        # 如果上下文包含姓名指示词，更可能是姓名
        if any(indicator in context for indicator in name_indicators):
            return True
        
        # 如果长度是2-3个字符且不在非姓名词汇中，可能是姓名
        if 2 <= len(value) <= 3:
            return True
        
        return False
    
    def _is_likely_english_name(self, value: str, text: str, start: int, end: int) -> bool:
        """判断英文字符串是否可能是姓名"""
        # 排除常见的非姓名词汇
        non_name_words = {
            'Foundation', 'Company', 'Corporation', 'Inc', 'Ltd', 'LLC', 'LLP',
            'Group', 'Holdings', 'Enterprises', 'Systems', 'Solutions', 'Services',
            'Technologies', 'International', 'Global', 'Worldwide', 'National',
            'Federal', 'State', 'Local', 'Regional', 'District', 'Department',
            'Bureau', 'Agency', 'Office', 'Center', 'Institute', 'University',
            'College', 'School', 'Hospital', 'Clinic', 'Bank', 'Credit', 'Union',
            'Insurance', 'Investment', 'Capital', 'Financial', 'Trading', 'Exchange',
            'Market', 'Store', 'Shop', 'Restaurant', 'Hotel', 'Resort', 'Club',
            'Association', 'Society', 'Organization', 'Committee', 'Council',
            'Board', 'Commission', 'Authority', 'Administration', 'Government'
        }
        
        # 检查是否包含非姓名词汇
        words = value.split()
        for word in words:
            if word in non_name_words:
                return False
        
        # 检查上下文，看是否在姓名相关的语境中
        context_start = max(0, start - 15)
        context_end = min(len(text), end + 15)
        context = text[context_start:context_end]
        
        # 姓名相关的关键词
        name_indicators = [
            'I am', 'My name is', 'I\'m', 'This is', 'Call me', 'I\'m called',
            'Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.', 'Sir', 'Madam', 'Miss',
            'CEO', 'President', 'Director', 'Manager', 'Officer', 'Agent',
            'Contact', 'Reach', 'Call', 'Email', 'Phone', 'Address'
        ]
        
        # 如果上下文包含姓名指示词，更可能是姓名
        if any(indicator.lower() in context.lower() for indicator in name_indicators):
            return True
        
        # 如果前面有称谓，更可能是姓名
        if start > 0:
            before_text = text[max(0, start-10):start].strip()
            if any(before_text.endswith(title) for title in ['Mr.', 'Mrs.', 'Ms.', 'Dr.', 'Prof.']):
                return True
        
        # 如果后面有称谓或标点，更可能是姓名
        if end < len(text):
            after_text = text[end:min(len(text), end+10)].strip()
            if any(after_text.startswith(title) for title in [',', '.', ' Jr.', ' Sr.', ' III', ' II']):
                return True
        
        # 如果长度合理且不在非姓名词汇中，可能是姓名
        if 2 <= len(words) <= 3:
            return True
        
        return False
    
    def _resolve_overlaps(self, entities: List[PIIEntity]) -> List[PIIEntity]:
        """解决位置重叠问题"""
        if not entities:
            return []
        
        # 按优先级排序
        sorted_entities = sorted(entities, key=lambda x: x.priority, reverse=True)
        
        filtered_entities = []
        used_positions = set()
        
        for entity in sorted_entities:
            start, end = entity.position
            
            # 检查重叠
            overlaps = any(pos in used_positions for pos in range(start, end))
            
            if not overlaps:
                for pos in range(start, end):
                    used_positions.add(pos)
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def _mask_value(self, pii_type: str, value: str) -> str:
        """脱敏处理"""
        if pii_type == 'email':
            parts = value.split('@')
            if len(parts) == 2:
                username = parts[0]
                domain = parts[1]
                if len(username) > 2:
                    masked_username = username[:2] + '*' * (len(username) - 2)
                else:
                    masked_username = '*' * len(username)
                return f"{masked_username}@{domain}"
        
        elif pii_type == 'phone':
            digits = re.sub(r'[^\d]', '', value)
            if len(digits) >= 7:
                return f"{digits[:3]}****{digits[-4:]}"
            else:
                return '*' * len(value)
        
        elif pii_type in ['chinese_name', 'english_name']:
            if len(value) > 1:
                return value[0] + '*' * (len(value) - 1)
            else:
                return '*'
        
        elif pii_type in ['id_card', 'bank_account']:
            if len(value) > 4:
                return value[0] + '*' * (len(value) - 2) + value[-1]
            else:
                return '*' * len(value)
        
        else:
            if len(value) > 2:
                return value[0] + '*' * (len(value) - 2) + value[-1]
            else:
                return '*' * len(value)


class FinalPIIProtector:
    """最终版本的PII保护器"""
    
    def __init__(self):
        self.detector = FinalPIIDetector()
        self.crypto = UserDataEncryption()
        self.field_crypto = FieldEncryption()
        self.pii_mapping = {}
    
    def protect_user_input(self, user_input: str, user_id: str = None, 
                          protection_level: str = "balanced") -> Dict[str, Any]:
        """保护用户输入"""
        # 检测PII
        pii_entities = self.detector.detect_pii(user_input)
        
        # 根据保护级别过滤
        if protection_level == "minimal":
            high_impact_types = {'id_card', 'bank_account', 'credit_card', 'ssn', 'phone', 'email'}
            filtered_entities = [e for e in pii_entities if e.type in high_impact_types]
        elif protection_level == "strict":
            filtered_entities = pii_entities
        else:  # balanced
            filtered_entities = []
            for entity in pii_entities:
                if entity.type in {'id_card', 'bank_account', 'credit_card', 'ssn', 'phone', 'email'}:
                    filtered_entities.append(entity)
                elif entity.type in {'chinese_name', 'english_name'}:
                    entity.masked_value = self._gentle_mask_name(entity.value, entity.type)
                    filtered_entities.append(entity)
        
        # 生成会话ID
        session_id = self._generate_session_id(user_input, user_id)
        
        # 加密PII数据
        encrypted_pii = []
        for entity in filtered_entities:
            entity.encrypted_value = self.field_crypto.encrypt_field(entity.value)
            encrypted_pii.append({
                'type': entity.type,
                'encrypted_value': entity.encrypted_value,
                'position': entity.position,
                'priority': entity.priority
            })
        
        # 创建匿名化文本
        anonymized_text = self._create_anonymized_text(user_input, filtered_entities, protection_level)
        
        # 存储PII映射
        if session_id not in self.pii_mapping:
            self.pii_mapping[session_id] = []
        self.pii_mapping[session_id].extend(encrypted_pii)
        
        return {
            'session_id': session_id,
            'original_text': user_input,
            'anonymized_text': anonymized_text,
            'pii_count': len(filtered_entities),
            'pii_types': list(set([entity.type for entity in filtered_entities])),
            'encrypted_pii': encrypted_pii,
            'protection_level': protection_level
        }
    
    def _gentle_mask_name(self, name: str, name_type: str) -> str:
        """温和的姓名脱敏"""
        if name_type == 'chinese_name':
            if len(name) > 1:
                return name[0] + '*' * (len(name) - 1)
            else:
                return name
        elif name_type == 'english_name':
            parts = name.split()
            if len(parts) >= 2:
                return f"{parts[0][0]}*** {parts[-1][0]}***"
            else:
                return name[0] + '*' * (len(name) - 1)
        return name
    
    def _create_anonymized_text(self, text: str, pii_entities: List[PIIEntity], protection_level: str) -> str:
        """创建匿名化文本"""
        # 按位置倒序排列
        sorted_entities = sorted(pii_entities, key=lambda x: x.position[0], reverse=True)
        
        anonymized_text = text
        for entity in sorted_entities:
            start, end = entity.position
            
            if protection_level == "minimal" and entity.type in {'chinese_name', 'english_name'}:
                continue
            elif protection_level == "balanced" and entity.type in {'chinese_name', 'english_name'}:
                replacement = f"[{entity.type.upper()}_GENTLE]"
            else:
                replacement = f"[{entity.type.upper()}_MASKED]"
            
            anonymized_text = (
                anonymized_text[:start] + 
                replacement + 
                anonymized_text[end:]
            )
        
        return anonymized_text
    
    def restore_pii(self, anonymized_text: str, session_id: str) -> str:
        """恢复PII信息 - 使用位置映射确保精确替换"""
        if session_id not in self.pii_mapping:
            return anonymized_text
        
        # 获取PII列表
        pii_list = self.pii_mapping[session_id]
        
        # 创建位置到PII的映射
        position_map = {}
        for pii in pii_list:
            start, end = pii['position']
            for pos in range(start, end):
                position_map[pos] = pii
        
        # 找到所有需要替换的标记
        markers = []
        for pii in pii_list:
            patterns = [
                f"[{pii['type'].upper()}_MASKED]",
                f"[{pii['type'].upper()}_GENTLE]"
            ]
            
            for pattern in patterns:
                if pattern in anonymized_text:
                    # 解密PII值
                    try:
                        decrypted_value = self.field_crypto.decrypt_field(pii['encrypted_value'])
                    except:
                        decrypted_value = f"[{pii['type'].upper()}_DECRYPT_FAILED]"
                    
                    markers.append((pattern, decrypted_value))
        
        # 执行替换
        restored_text = anonymized_text
        for pattern, replacement in markers:
            restored_text = restored_text.replace(pattern, replacement)
        
        return restored_text
    
    def _generate_session_id(self, text: str, user_id: str = None) -> str:
        """生成会话ID"""
        if user_id:
            content = f"{user_id}_{text}_{uuid.uuid4()}"
        else:
            content = f"{text}_{uuid.uuid4()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


def demo_final_pii_protection():
    """演示最终版本的PII保护"""
    print("🎯 最终版本PII保护演示")
    print("=" * 80)
    
    protector = FinalPIIProtector()
    
    test_cases = [
        {
            'input': '我叫张三，电话13812345678，邮箱zhangsan@company.com，请帮我分析AAPL',
            'description': '中文姓名+电话+邮箱+投资查询'
        },
        {
            'input': 'Hi, I am John Smith, phone +1-555-123-4567, email john@example.com, help with portfolio',
            'description': '英文姓名+电话+邮箱+投资组合'
        },
        {
            'input': '我是李小明，身份证110101199001011234，银行卡6222021234567890，想了解ESG投资',
            'description': '中文姓名+身份证+银行卡+ESG投资'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {case['description']}")
        print(f"原始输入: {case['input']}")
        
        # 保护PII
        result = protector.protect_user_input(case['input'], f"user_{i}", "balanced")
        
        print(f"匿名化文本: {result['anonymized_text']}")
        print(f"PII保护: {result['pii_count']} 个 - {result['pii_types']}")
        
        # 模拟LLM响应
        llm_response = f"我收到了您的请求：{result['anonymized_text']}"
        
        # 恢复PII
        restored = protector.restore_pii(llm_response, result['session_id'])
        print(f"恢复后: {restored}")
        print("-" * 60)


if __name__ == "__main__":
    demo_final_pii_protection()
