"""
智能PII保护策略
优化PII保护对LLM使用的影响
"""

import re
from typing import Dict, List, Any, Optional
from final_pii_protection import FinalPIIProtector, FinalPIIDetector


class SmartPIIProtector(FinalPIIProtector):
    """智能PII保护器 - 优化对LLM使用的影响"""
    
    def __init__(self):
        super().__init__()
        # 定义对LLM理解影响较小的PII类型
        self.low_impact_pii = {'chinese_name', 'english_name'}
        # 定义对LLM理解影响较大的PII类型
        self.high_impact_pii = {'email', 'phone', 'ssn', 'credit_card', 'bank_account', 'id_card'}
    
    def smart_protect_user_input(self, user_input: str, user_id: str = None, 
                                protection_level: str = "balanced") -> Dict[str, Any]:
        """
        智能保护用户输入 - 根据保护级别调整策略
        
        Args:
            user_input: 用户原始输入
            user_id: 用户ID
            protection_level: 保护级别 ("minimal", "balanced", "strict")
            
        Returns:
            保护后的数据
        """
        # 检测PII
        pii_entities = self.detector.detect_pii(user_input)
        
        # 根据保护级别过滤PII
        if protection_level == "minimal":
            # 最小保护：只保护高敏感PII
            filtered_entities = [e for e in pii_entities if e.type in self.high_impact_pii]
        elif protection_level == "strict":
            # 严格保护：保护所有PII
            filtered_entities = pii_entities
        else:  # balanced
            # 平衡保护：保护高敏感PII，对姓名类PII使用更温和的处理
            filtered_entities = []
            for entity in pii_entities:
                if entity.type in self.high_impact_pii:
                    filtered_entities.append(entity)
                elif entity.type in self.low_impact_pii:
                    # 对姓名使用更温和的脱敏
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
                'position': entity.position
            })
        
        # 创建智能匿名化文本
        anonymized_text = self._create_smart_anonymized_text(user_input, filtered_entities, protection_level)
        
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
    
    def _create_smart_anonymized_text(self, text: str, pii_entities: List, protection_level: str) -> str:
        """创建智能匿名化文本"""
        # 按位置倒序排列
        sorted_entities = sorted(pii_entities, key=lambda x: x.position[0], reverse=True)
        
        anonymized_text = text
        for entity in sorted_entities:
            start, end = entity.position
            
            if protection_level == "minimal" and entity.type in self.low_impact_pii:
                # 最小保护模式下，保留姓名信息
                continue
            elif protection_level == "balanced" and entity.type in self.low_impact_pii:
                # 平衡模式下，使用温和的标记
                replacement = f"[{entity.type.upper()}_GENTLE]"
            else:
                # 严格模式下，使用标准标记
                replacement = f"[{entity.type.upper()}_MASKED]"
            
            anonymized_text = (
                anonymized_text[:start] + 
                replacement + 
                anonymized_text[end:]
            )
        
        return anonymized_text
    
    def restore_smart_pii(self, anonymized_text: str, session_id: str) -> str:
        """智能恢复PII"""
        if session_id not in self.pii_mapping:
            return anonymized_text
        
        restored_text = anonymized_text
        pii_list = self.pii_mapping[session_id]
        
        # 按位置倒序排列
        sorted_pii = sorted(pii_list, key=lambda x: x['position'][0], reverse=True)
        
        for pii in sorted_pii:
            # 解密PII值
            try:
                decrypted_value = self.field_crypto.decrypt_field(pii['encrypted_value'])
            except:
                decrypted_value = f"[{pii['type'].upper()}_DECRYPT_FAILED]"
            
            # 替换各种匿名化标记
            patterns = [
                f"[{pii['type'].upper()}_MASKED]",
                f"[{pii['type'].upper()}_GENTLE]"
            ]
            
            for pattern in patterns:
                if pattern in restored_text:
                    restored_text = restored_text.replace(pattern, decrypted_value, 1)
                    break
        
        return restored_text


def demo_smart_protection():
    """演示智能PII保护"""
    print("🧠 智能PII保护演示")
    print("=" * 80)
    
    protector = SmartPIIProtector()
    
    test_input = "我叫张三，电话13812345678，邮箱zhangsan@company.com，请帮我分析AAPL的投资价值"
    
    protection_levels = ["minimal", "balanced", "strict"]
    
    for level in protection_levels:
        print(f"\n🔒 保护级别: {level.upper()}")
        print("-" * 60)
        
        result = protector.smart_protect_user_input(test_input, "test_user", level)
        
        print(f"原始输入: {test_input}")
        print(f"匿名化文本: {result['anonymized_text']}")
        print(f"检测到PII: {result['pii_count']} 个")
        print(f"PII类型: {', '.join(result['pii_types'])}")
        
        # 模拟LLM处理
        llm_input = result['anonymized_text']
        print(f"发送给LLM: {llm_input}")
        
        # 恢复PII
        restored = protector.restore_smart_pii(llm_input, result['session_id'])
        print(f"恢复后: {restored}")


def analyze_llm_impact():
    """分析PII保护对LLM的影响"""
    print("\n📊 PII保护对LLM影响分析")
    print("=" * 80)
    
    protector = SmartPIIProtector()
    
    test_cases = [
        {
            "input": "我叫张三，请帮我分析AAPL的投资价值",
            "description": "包含姓名 + 投资查询"
        },
        {
            "input": "我的电话是13812345678，邮箱zhangsan@company.com，想了解ESG投资",
            "description": "包含联系方式 + ESG投资"
        },
        {
            "input": "请帮我查询AAPL的股价",
            "description": "纯投资查询（无PII）"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 测试用例 {i}: {case['description']}")
        print("-" * 60)
        
        # 测试不同保护级别
        for level in ["minimal", "balanced", "strict"]:
            result = protector.smart_protect_user_input(case['input'], f"user_{i}", level)
            
            # 分析文本可读性
            anonymized = result['anonymized_text']
            readability_score = calculate_readability_score(anonymized)
            
            print(f"{level.upper()} 保护:")
            print(f"  匿名化文本: {anonymized}")
            print(f"  可读性评分: {readability_score}/10")
            print(f"  PII保护数量: {result['pii_count']}")


def calculate_readability_score(text: str) -> int:
    """计算文本可读性评分（1-10）"""
    score = 10
    
    # 减少分数的因素
    if '[MASKED]' in text:
        score -= 2
    if '[GENTLE]' in text:
        score -= 1
    if text.count('[') > 3:
        score -= 1
    if len(text) < 10:
        score -= 1
    
    return max(1, score)


if __name__ == "__main__":
    demo_smart_protection()
    analyze_llm_impact()
