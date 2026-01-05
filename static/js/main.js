// 预测按钮点击事件
document.getElementById('predict-btn').addEventListener('click', async () => {
    const text = document.getElementById('text-input').value;
    if (!text) {
        alert('请输入文本');
        return;
    }

    // 调用后端预测接口
    const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    });

    const result = await response.json();
    const resultContent = document.getElementById('result-content');

    // 展示预测结果（修复字段名 + 添加NaN校验）
    // 1. 提取置信度并处理异常值
    const confidence = result.confidence; // 关键修复：字段名从score改为confidence
    let confidencePercent;
    if (typeof confidence === 'number' && !isNaN(confidence)) {
        confidencePercent = (confidence * 100).toFixed(2) + '%';
    } else {
        confidencePercent = '未知（模型输出异常）';
    }

    // 2. 渲染结果
    if (result.label === 1) {
        resultContent.innerHTML = `
            <p>情感类别：正面情感</p>
            <p>置信度：${confidencePercent}</p>
        `;
        resultContent.style.backgroundColor = '#e8f5e9';
    } else {
        resultContent.innerHTML = `
            <p>情感类别：负面情感</p>
            <p>置信度：${confidencePercent}</p>
        `;
        resultContent.style.backgroundColor = '#ffebee';
    }
});

// 页面加载时渲染模型性能图表（保持不变）
window.onload = () => {
    // 训练损失曲线
    const lossCtx = document.getElementById('loss-chart').getContext('2d');
    new Chart(lossCtx, {
        type: 'line',
        data: {
            labels: [0, 50, 100, 200, 300, 400, 550, 700, 850, 1400, 1600],
            datasets: [
                { label: '训练损失', data: [0.8861, 0.8528, 1.2711, 0.8514, 0.6984, 0.7083, 0.5597, 0.3531, 0.7113, 0.5268, 0.6426], borderColor: '#FF6B6B' },
                { label: '验证损失', data: [0.9829, 0.9106, 0.7730, 0.7249, 0.6878, 0.6565, 0.6226, 0.6150, 0.6620, 0.5685, 0.5692], borderColor: '#4ECDC4' }
            ]
        },
        options: { responsive: true, title: { display: true, text: '训练收敛曲线' } }
    });

    // 混淆矩阵
    const confusionCtx = document.getElementById('confusion-chart').getContext('2d');
    new Chart(confusionCtx, {
        type: 'bar',
        data: {
            labels: ['负面情感（预测）', '正面情感（预测）'],
            datasets: [
                { label: '负面情感（真实）', data: [3561, 2422], backgroundColor: '#FF9F43' },
                { label: '正面情感（真实）', data: [1242, 4710], backgroundColor: '#10AC84' }
            ]
        },
        options: { responsive: true, title: { display: true, text: '混淆矩阵' }, scales: { x: { stacked: true }, y: { stacked: true } } }
    });
};