from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import torch
import numpy as np
import os
from datetime import datetime
from models.bert_target import Config, Model  # 确保该路径正确

# ==================== 1. 初始化 Flask 应用 ====================
app = Flask(__name__)
# 配置密钥（必须修改为自己的随机字符串，用于加密会话）
app.config['SECRET_KEY'] = 'emotion-classification-secret-key-2026'
# 配置数据库路径：使用项目根目录下的 emotion.db，避免 instance 目录权限问题
db_path = os.path.join(app.root_path, 'emotion.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # 关闭不必要的警告

# ==================== 2. 初始化数据库和登录管理器 ====================
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # 未登录时重定向的端点
# 新增：未授权访问回调，确保强制跳转到登录页
@login_manager.unauthorized_handler
def unauthorized():
    return redirect(url_for('login'))

# ==================== 3. 定义数据库模型（必须在 db 初始化之后） ====================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)
    records = db.relationship('PredictRecord', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class PredictRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text, nullable=False)
    label_name = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    create_time = db.Column(db.DateTime, default=datetime.now)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# ==================== 4. Flask-Login 回调函数 ====================
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==================== 5. 加载情感分析模型（可选：放在路由前） ====================
print("开始加载情感分析模型...")
config = Config()
model = Model(config)
model_path = os.path.join(config.project_root, "saved_dict", "bert_target.ckpt")
print(f"模型路径：{model_path}，是否存在：{os.path.exists(model_path)}")

# 加载模型权重
try:
    state_dict = torch.load(model_path, map_location=config.device)
    # 处理多GPU训练的权重前缀
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败：{e}")
    exit(1)  # 模型加载失败则退出服务

# ==================== 6. 定义路由（核心：必须在所有初始化之后） ====================
@app.route('/')
@login_required
def index():
    return render_template('index.html', username=current_user.username)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('请输入用户名和密码')
            return redirect(url_for('login'))
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('用户名或密码错误')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            flash('请输入用户名和密码')
            return redirect(url_for('register'))
        if User.query.filter_by(username=username).first():
            flash('用户名已存在')
            return redirect(url_for('register'))
        # 创建新用户
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        try:
            db.session.commit()
            flash('注册成功，请登录')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash(f'注册失败：{e}')
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('已成功退出登录')
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        if not text:
            return jsonify({'error': '请输入有效文本'}), 400

        # 数据预处理
        token = config.tokenizer.tokenize(text)
        token = ['[CLS]'] + token + ['[SEP]']
        token_ids = config.tokenizer.convert_tokens_to_ids(token)
        seq_len = len(token_ids)
        mask = [1] * seq_len

        # 截断或填充到指定长度
        if seq_len < config.pad_size:
            token_ids += [0] * (config.pad_size - seq_len)
            mask += [0] * (config.pad_size - seq_len)
        else:
            token_ids = token_ids[:config.pad_size]
            mask = mask[:config.pad_size]
            seq_len = config.pad_size

        # 转换为 tensor
        input_ids = torch.tensor([token_ids], dtype=torch.long).to(config.device)
        seq_len_tensor = torch.tensor([seq_len], dtype=torch.long).to(config.device)
        mask_tensor = torch.tensor([mask], dtype=torch.long).to(config.device)

        # 模型预测
        with torch.no_grad():
            outputs = model((input_ids, seq_len_tensor, mask_tensor))
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            label_idx = np.argmax(probabilities)
            score = round(float(probabilities[label_idx]), 4)
            label_name = config.class_list[label_idx]

        # 保存预测记录到数据库
        new_record = PredictRecord(
            text=text,
            label_name=label_name,
            confidence=score,
            user_id=current_user.id
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({
            'text': text,
            'label_name': label_name,
            'label': int(label_idx),
            'confidence': score
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/records')
@login_required
def records():
    user_records = PredictRecord.query.filter_by(user_id=current_user.id).order_by(PredictRecord.create_time.desc()).all()
    return render_template('records.html', records=user_records)

# ==================== 7. 初始化数据库（仅执行一次） ====================
with app.app_context():
    db.create_all()
    print(f"✅ 数据库初始化成功，文件路径：{db_path}")

# ==================== 8. 启动服务 ====================
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)