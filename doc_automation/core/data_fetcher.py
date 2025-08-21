"""
数据获取器模块
从各种数据源获取审核相关数据
"""

import random
from datetime import datetime, timedelta
from typing import Dict, Any


class DataFetcher:
    def __init__(self):
        self.mock_data = True  # 生产环境中应连接真实数据源
    
    def get_audit_metrics(self) -> Dict[str, Any]:
        """获取审核性能指标"""
        if self.mock_data:
            return {
                'total_processed': random.randint(10000, 50000),
                'accuracy_rate': round(random.uniform(0.92, 0.98), 3),
                'false_positive_rate': round(random.uniform(0.01, 0.03), 3),
                'false_negative_rate': round(random.uniform(0.005, 0.015), 3),
                'avg_processing_time': f"{random.randint(15, 45)}s",
                'escalation_ratio': round(random.uniform(0.02, 0.08), 3),
                'daily_trends': [
                    {'date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                     'volume': random.randint(1000, 3000),
                     'accuracy': round(random.uniform(0.90, 0.98), 3)}
                    for i in range(7)
                ]
            }
        
        # 这里应该连接实际的数据库或 API
        return {}
    
    def get_experiment_data(self) -> Dict[str, Any]:
        """获取实验数据"""
        if self.mock_data:
            return {
                'ab_tests': [
                    {
                        'experiment_id': 'exp_001',
                        'name': 'Prompt优化实验',
                        'start_date': '2024-01-01',
                        'end_date': '2024-01-15',
                        'groups': {
                            'control': {'accuracy': 0.92, 'latency': 45},
                            'treatment': {'accuracy': 0.95, 'latency': 42}
                        },
                        'statistical_significance': 0.95
                    }
                ],
                'model_versions': [
                    {
                        'version': 'v2.1.0',
                        'release_date': '2024-01-10',
                        'improvements': ['提升准确率3%', '降低延迟10%'],
                        'performance_delta': {'accuracy': 0.03, 'latency': -0.1}
                    }
                ]
            }
        
        return {}
    
    def get_policy_configs(self) -> Dict[str, Any]:
        """获取策略配置数据"""
        if self.mock_data:
            return {
                'risk_categories': [
                    {'name': '暴力内容', 'threshold': 0.8, 'action': 'block'},
                    {'name': '色情内容', 'threshold': 0.7, 'action': 'review'},
                    {'name': '虚假信息', 'threshold': 0.6, 'action': 'flag'}
                ],
                'content_types': [
                    {'type': '短视频', 'daily_volume': 50000, 'auto_rate': 0.85},
                    {'type': '图文', 'daily_volume': 30000, 'auto_rate': 0.90},
                    {'type': '直播', 'daily_volume': 5000, 'auto_rate': 0.70}
                ],
                'escalation_rules': [
                    {'condition': 'confidence < 0.5', 'action': 'human_review'},
                    {'condition': 'risk_score > 0.9', 'action': 'expert_review'}
                ]
            }
        
        return {}
    
    def get_user_feedback(self) -> Dict[str, Any]:
        """获取用户反馈数据"""
        if self.mock_data:
            return {
                'satisfaction_scores': [
                    {'department': '内容安全', 'score': 4.2, 'responses': 156},
                    {'department': '产品运营', 'score': 3.8, 'responses': 89},
                    {'department': '算法团队', 'score': 4.5, 'responses': 67}
                ],
                'feature_requests': [
                    {'feature': '批量审核', 'votes': 23, 'priority': 'high'},
                    {'feature': '自定义规则', 'votes': 18, 'priority': 'medium'},
                    {'feature': '移动端支持', 'votes': 12, 'priority': 'low'}
                ],
                'pain_points': [
                    '审核界面响应慢',
                    '误报率偏高',
                    '培训材料不足'
                ]
            }
        
        return {}
