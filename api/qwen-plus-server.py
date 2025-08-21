#!/usr/bin/env python3
"""
qwen-plus推理服务器
为前端提供智能决策API
"""

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import dashscope
from dashscope import Generation
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # 允许跨域请求

dashscope.api_key = os.getenv('DASHSCOPE_API_KEY', 'sk-placeholder-replace-with-real-key')

@app.route('/api/qwen-plus-reasoning', methods=['POST'])
def qwen_plus_reasoning():
    """
    qwen-plus文本推理API
    """
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 1000)
        temperature = data.get('temperature', 0.7)
        
        if not prompt:
            return jsonify({'error': '缺少prompt参数'}), 400
        
        logger.info(f"🧠 qwen-plus推理请求: {prompt[:100]}...")
        
        response = Generation.call(
            model='qwen-plus',
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.8,
            repetition_penalty=1.1
        )
        
        if response.status_code == 200:
            output_text = response.output.text
            logger.info(f"✅ qwen-plus推理成功: {output_text[:100]}...")
            
            return jsonify({
                'choices': [{
                    'message': {
                        'content': output_text
                    }
                }],
                'output': {
                    'text': output_text
                },
                'usage': {
                    'prompt_tokens': len(prompt) // 4,
                    'completion_tokens': len(output_text) // 4,
                    'total_tokens': (len(prompt) + len(output_text)) // 4
                }
            })
        else:
            logger.error(f"❌ qwen-plus调用失败: {response}")
            return jsonify({'error': f'模型调用失败: {response.message}'}), 500
            
    except Exception as e:
        logger.error(f"❌ qwen-plus服务器错误: {str(e)}")
        return jsonify({'error': f'服务器错误: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({'status': 'healthy', 'service': 'qwen-plus-reasoning'})

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5002))
    print(f"🚀 qwen-plus推理服务器启动在端口 {port}")
    print(f"📍 API端点: http://localhost:{port}/api/qwen-plus-reasoning")
    app.run(host='0.0.0.0', port=port, debug=True)
