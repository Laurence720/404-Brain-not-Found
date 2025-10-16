"""
用户数据加密工具类
提供多种加密方案用于保护用户敏感数据
"""

import os
import json
import base64
import secrets
from typing import Dict, Any, Optional, Union
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import hashlib


class CryptoManager:
    """加密管理器 - 提供多种加密方案"""
    
    def __init__(self, master_key: Optional[str] = None):
        """
        初始化加密管理器
        
        Args:
            master_key: 主密钥，如果未提供则从环境变量获取
        """
        # 尝试加载环境变量
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        
        self.master_key = master_key or os.getenv('CRYPTO_MASTER_KEY')
        if not self.master_key:
            import secrets
            self.master_key = secrets.token_urlsafe(48)
            print("[crypto-warning] CRYPTO_MASTER_KEY not set; generated ephemeral key for this session.")
        
        # 生成派生密钥
        self.derived_key = self._derive_key(self.master_key)
        self.fernet = Fernet(self.derived_key)
    
    def _derive_key(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """使用PBKDF2派生密钥"""
        if salt is None:
            salt = b'wx_langgraph_salt_2024'  # 生产环境中应使用随机盐
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt_data(self, data: Union[str, Dict, Any]) -> str:
        """
        加密数据（使用AES-256-GCM）
        
        Args:
            data: 要加密的数据
            
        Returns:
            加密后的base64字符串
        """
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False)
        
        data_bytes = data.encode('utf-8')
        encrypted_data = self.fernet.encrypt(data_bytes)
        return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> Union[str, Dict, Any]:
        """
        解密数据
        
        Args:
            encrypted_data: 加密的base64字符串
            
        Returns:
            解密后的数据
        """
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
            decrypted_bytes = self.fernet.decrypt(encrypted_bytes)
            decrypted_str = decrypted_bytes.decode('utf-8')
            
            # 尝试解析为JSON
            try:
                return json.loads(decrypted_str)
            except json.JSONDecodeError:
                return decrypted_str
        except Exception as e:
            raise ValueError(f"解密失败: {e}")
    
    def encrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        加密文件
        
        Args:
            file_path: 要加密的文件路径
            output_path: 输出文件路径，默认为原文件+.enc
            
        Returns:
            加密后的文件路径
        """
        if output_path is None:
            output_path = file_path + '.enc'
        
        with open(file_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.fernet.encrypt(data)
        
        with open(output_path, 'wb') as f:
            f.write(encrypted_data)
        
        return output_path
    
    def decrypt_file(self, encrypted_file_path: str, output_path: Optional[str] = None) -> str:
        """
        解密文件
        
        Args:
            encrypted_file_path: 加密文件路径
            output_path: 输出文件路径，默认为去掉.enc后缀
            
        Returns:
            解密后的文件路径
        """
        if output_path is None:
            output_path = encrypted_file_path.replace('.enc', '')
        
        with open(encrypted_file_path, 'rb') as f:
            encrypted_data = f.read()
        
        decrypted_data = self.fernet.decrypt(encrypted_data)
        
        with open(output_path, 'wb') as f:
            f.write(decrypted_data)
        
        return output_path


class FieldEncryption:
    """字段级加密 - 用于加密特定敏感字段"""
    
    def __init__(self, key: Optional[bytes] = None):
        if key is None:
            # 使用固定的密钥派生，确保加密解密一致性
            import hashlib
            master_key = os.getenv('CRYPTO_MASTER_KEY', 'default_key')
            self.key = hashlib.sha256(master_key.encode()).digest()
        else:
            self.key = key
    
    def encrypt_field(self, value: str) -> str:
        """加密单个字段"""
        if not value:
            return value
        
        # 生成随机IV
        iv = secrets.token_bytes(16)
        
        # 使用AES-256-CBC加密
        cipher = Cipher(
            algorithms.AES(self.key),
            modes.CBC(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        
        # PKCS7填充
        value_bytes = value.encode('utf-8')
        pad_length = 16 - (len(value_bytes) % 16)
        padded_value = value_bytes + bytes([pad_length] * pad_length)
        
        encrypted = encryptor.update(padded_value) + encryptor.finalize()
        
        # 返回IV+加密数据的base64编码
        return base64.urlsafe_b64encode(iv + encrypted).decode()
    
    def decrypt_field(self, encrypted_value: str) -> str:
        """解密单个字段"""
        if not encrypted_value:
            return encrypted_value
        
        try:
            data = base64.urlsafe_b64decode(encrypted_value.encode())
            iv = data[:16]
            encrypted = data[16:]
            
            cipher = Cipher(
                algorithms.AES(self.key),
                modes.CBC(iv),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            decrypted = decryptor.update(encrypted) + decryptor.finalize()
            
            # 去除PKCS7填充
            pad_length = decrypted[-1]
            if pad_length > 0 and pad_length <= 16:
                return decrypted[:-pad_length].decode('utf-8')
            else:
                return decrypted.decode('utf-8')
        except Exception as e:
            raise ValueError(f"字段解密失败: {e}")


class UserDataEncryption:
    """用户数据加密管理器"""
    
    def __init__(self):
        self.crypto_manager = CryptoManager()
        self.field_encryption = FieldEncryption()
    
    def encrypt_user_profile(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        加密用户配置文件
        
        Args:
            profile: 用户配置文件
            
        Returns:
            加密后的配置文件
        """
        # 定义需要加密的敏感字段
        sensitive_fields = [
            'user_id', 'email', 'phone', 'ssn', 'bank_account',
            'investment_goals', 'financial_info'
        ]
        
        encrypted_profile = profile.copy()
        
        for field in sensitive_fields:
            if field in encrypted_profile and encrypted_profile[field]:
                encrypted_profile[field] = self.field_encryption.encrypt_field(
                    str(encrypted_profile[field])
                )
        
        return encrypted_profile
    
    def decrypt_user_profile(self, encrypted_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        解密用户配置文件
        
        Args:
            encrypted_profile: 加密的配置文件
            
        Returns:
            解密后的配置文件
        """
        sensitive_fields = [
            'user_id', 'email', 'phone', 'ssn', 'bank_account',
            'investment_goals', 'financial_info'
        ]
        
        decrypted_profile = encrypted_profile.copy()
        
        for field in sensitive_fields:
            if field in decrypted_profile and decrypted_profile[field] is not None and str(decrypted_profile[field]).strip():
                try:
                    decrypted_profile[field] = self.field_encryption.decrypt_field(
                        str(decrypted_profile[field])
                    )
                except ValueError:
                    # 如果解密失败，保持原值（可能是未加密的数据）
                    pass
        
        return decrypted_profile
    
    def save_encrypted_profile(self, user_id: str, profile: Dict[str, Any], 
                             base_dir: str = "./data/user_profiles") -> str:
        """
        保存加密的用户配置文件
        
        Args:
            user_id: 用户ID
            profile: 用户配置文件
            base_dir: 基础目录
            
        Returns:
            保存的文件路径
        """
        os.makedirs(base_dir, exist_ok=True)
        
        # 加密配置文件
        encrypted_profile = self.encrypt_user_profile(profile)
        
        # 保存到文件
        file_path = os.path.join(base_dir, f"{user_id}.json.enc")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(encrypted_profile, f, ensure_ascii=False, indent=2)
        
        return file_path
    
    def load_encrypted_profile(self, user_id: str, 
                             base_dir: str = "./data/user_profiles") -> Dict[str, Any]:
        """
        加载并解密用户配置文件
        
        Args:
            user_id: 用户ID
            base_dir: 基础目录
            
        Returns:
            解密后的配置文件
        """
        file_path = os.path.join(base_dir, f"{user_id}.json.enc")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"用户配置文件不存在: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            encrypted_profile = json.load(f)
        
        return self.decrypt_user_profile(encrypted_profile)


def generate_master_key() -> str:
    """生成主密钥"""
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()


def setup_encryption_env():
    """设置加密环境变量"""
    env_file = ".env"
    
    if not os.path.exists(env_file):
        print(f"创建 {env_file} 文件...")
    
    # 检查是否已有加密密钥
    with open(env_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'CRYPTO_MASTER_KEY' not in content:
        master_key = generate_master_key()
        with open(env_file, 'a', encoding='utf-8') as f:
            f.write(f"\n# 数据加密主密钥\nCRYPTO_MASTER_KEY={master_key}\n")
        print(f"已生成加密主密钥并添加到 {env_file}")
    else:
        print("加密主密钥已存在")


if __name__ == "__main__":
    # 示例用法
    setup_encryption_env()
    
    # 创建加密管理器
    crypto = UserDataEncryption()
    
    # 示例用户数据
    sample_profile = {
        "user_id": "user123",
        "email": "user@example.com",
        "risk_level": "medium",
        "preferences": {
            "esg": True,
            "regions": ["US", "EU"],
            "sectors": ["technology", "healthcare"]
        },
        "financial_info": {
            "annual_income": 100000,
            "investment_goals": "retirement"
        }
    }
    
    # 加密并保存
    encrypted_file = crypto.save_encrypted_profile("user123", sample_profile)
    print(f"加密配置文件已保存到: {encrypted_file}")
    
    # 加载并解密
    decrypted_profile = crypto.load_encrypted_profile("user123")
    print("解密后的配置文件:")
    print(json.dumps(decrypted_profile, ensure_ascii=False, indent=2))
