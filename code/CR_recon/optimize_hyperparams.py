"""
Hyperparameter Optimizer: Train log를 분석하고 더 나은 설정을 찾아줌
사용: python optimize_hyperparams.py --log outputs/train_log.txt
"""
import argparse
import re
import yaml
from pathlib import Path
from typing import Dict, List, Tuple
import subprocess
import json


def parse_train_log(log_path: str) -> Dict:
    """Train log를 파싱해서 설정과 결과 추출"""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    result = {
        'model_name': None,
        'model_path': None,
        'model_params': {},
        'loss_name': None,
        'loss_path': None,
        'loss_params': {},
        'hyperparams': {},
        'best_val_loss': float('inf'),
        'final_val_loss': float('inf'),
        'epochs_trained': 0,
    }

    # Model 정보 추출
    model_match = re.search(r'Model: (\w+)', content)
    if model_match:
        result['model_name'] = model_match.group(1)

    model_path_match = re.search(r'Model Path: ([\w/._-]+)', content)
    if model_path_match:
        result['model_path'] = model_path_match.group(1)

    # Model Parameters 추출
    model_params_section = re.search(
        r'Model Parameters:(.*?)(?=Loss Function|Hyperparameters)',
        content,
        re.DOTALL
    )
    if model_params_section:
        params_text = model_params_section.group(1)
        for line in params_text.split('\n'):
            match = re.search(r'-\s*(\w+):\s*(.*)', line)
            if match:
                param_name = match.group(1)
                param_value = match.group(2).strip()
                try:
                    result['model_params'][param_name] = yaml.safe_load(param_value)
                except:
                    result['model_params'][param_name] = param_value

    # Loss 정보 추출
    loss_match = re.search(r'Loss Function: (\w+)', content)
    if loss_match:
        result['loss_name'] = loss_match.group(1)

    loss_path_match = re.search(r'Loss Path: ([\w/._-]+)', content)
    if loss_path_match:
        result['loss_path'] = loss_path_match.group(1)

    # Loss Parameters 추출
    loss_params_section = re.search(
        r'Loss Parameters:(.*?)(?=Hyperparameters)',
        content,
        re.DOTALL
    )
    if loss_params_section:
        params_text = loss_params_section.group(1)
        for line in params_text.split('\n'):
            match = re.search(r'-\s*(\w+):\s*(.*)', line)
            if match:
                param_name = match.group(1)
                param_value = match.group(2).strip()
                try:
                    result['loss_params'][param_name] = yaml.safe_load(param_value)
                except:
                    result['loss_params'][param_name] = param_value

    # Hyperparameters 추출
    hyper_section = re.search(
        r'Hyperparameters:(.*?)(?=Data Statistics)',
        content,
        re.DOTALL
    )
    if hyper_section:
        hyper_text = hyper_section.group(1)
        for line in hyper_text.split('\n'):
            match = re.search(r'-\s*([\w\s]+):\s*(.*)', line)
            if match:
                param_name = match.group(1).strip()
                param_value = match.group(2).strip()
                try:
                    result['hyperparams'][param_name] = yaml.safe_load(param_value)
                except:
                    result['hyperparams'][param_name] = param_value

    # Best val loss 추출
    best_loss_matches = re.findall(
        r'best_val=([\d.e+-]+)',
        content
    )
    if best_loss_matches:
        result['best_val_loss'] = float(best_loss_matches[-1])
        result['final_val_loss'] = float(best_loss_matches[-1])

    # Epochs trained 추출
    epoch_matches = re.findall(
        r'\[EPOCH\]\s+(\d+)/(\d+)',
        content
    )
    if epoch_matches:
        result['epochs_trained'] = int(epoch_matches[-1][0])

    return result


def generate_suggestions(log_data: Dict) -> List[Dict]:
    """로그 분석 후 개선 제안 생성"""
    suggestions = []

    print("\n" + "=" * 80)
    print("ANALYSIS & SUGGESTIONS")
    print("=" * 80)

    print(f"\n📊 Current Results:")
    print(f"  - Model: {log_data['model_name']}")
    print(f"  - Loss: {log_data['loss_name']}")
    print(f"  - Best Val Loss: {log_data['best_val_loss']:.6e}")
    print(f"  - Epochs Trained: {log_data['epochs_trained']}")

    # 1. Learning Rate 조정
    current_lr = log_data['hyperparams'].get('Learning Rate', 0.001)
    if log_data['best_val_loss'] > 0.05:
        suggestions.append({
            'name': 'Increase Learning Rate',
            'description': f'Loss가 높으니 Learning Rate를 {current_lr} → {current_lr * 2} 로 증가',
            'config_changes': {'training': {'lr': current_lr * 2}},
            'reason': 'High loss indicates slow convergence'
        })
    elif log_data['best_val_loss'] < 0.01 and log_data['epochs_trained'] > 100:
        suggestions.append({
            'name': 'Decrease Learning Rate',
            'description': f'Loss가 매우 낮으니 미세 조정: Learning Rate {current_lr} → {current_lr * 0.5}',
            'config_changes': {'training': {'lr': current_lr * 0.5}},
            'reason': 'Fine-tuning for better convergence'
        })

    # 2. Model 변경 제안
    if log_data['model_name'] == 'cnn_xattn':
        suggestions.append({
            'name': 'Try CNN_GRU Model',
            'description': 'CNN_XAttn 대신 CNN_GRU 모델로 비교 (간단한 baseline)',
            'config_changes': {'model': {'name': 'cnn_gru'}},
            'reason': 'More lightweight, faster convergence'
        })
    elif log_data['model_name'] == 'cnn_gru':
        suggestions.append({
            'name': 'Try CNN_XAttn Model',
            'description': 'CNN_GRU 대신 CNN_XAttn 모델로 비교 (더 강력함)',
            'config_changes': {'model': {'name': 'cnn_xattn'}},
            'reason': 'More expressive model with attention'
        })

    # 3. Loss Function 변경 제안
    if log_data['loss_name'] == 'mse_pearson':
        suggestions.append({
            'name': 'Try Weighted Smooth Loss',
            'description': 'MSE_Pearson 대신 Weighted_Smooth Loss로 비교 (smoothness 정규화)',
            'config_changes': {'loss': {'name': 'weighted_smooth'}},
            'reason': 'Better for smooth spectral predictions'
        })
    elif log_data['loss_name'] == 'weighted_smooth':
        suggestions.append({
            'name': 'Try MSE_Pearson Loss',
            'description': 'Weighted_Smooth 대신 MSE_Pearson Loss로 비교',
            'config_changes': {'loss': {'name': 'mse_pearson'}},
            'reason': 'Scale/shift invariant, simpler'
        })

    # 4. Weight Decay 조정
    wd = log_data['hyperparams'].get('Weight Decay', 0.005)
    if log_data['best_val_loss'] < 0.01:
        suggestions.append({
            'name': 'Increase Weight Decay',
            'description': f'Regularization 증가: {wd} → {wd * 2}',
            'config_changes': {'training': {'weight_decay': wd * 2}},
            'reason': 'Overfitting prevention'
        })

    # 5. Batch Size 조정
    bs = log_data['hyperparams'].get('Batch Size', 64)
    if log_data['best_val_loss'] > 0.02:
        suggestions.append({
            'name': 'Increase Batch Size',
            'description': f'Batch Size 증가: {bs} → {bs * 2} (안정성 향상)',
            'config_changes': {'data': {'batch_size': bs * 2}},
            'reason': 'Stable gradient estimates'
        })

    return suggestions


def create_test_configs(base_config_path: str, suggestions: List[Dict]) -> List[Tuple[str, Dict]]:
    """각 제안에 대해 테스트 config 생성"""
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    test_configs = []

    for i, suggestion in enumerate(suggestions):
        config = yaml.safe_load(yaml.dump(base_config))

        # 테스트용으로 epoch 줄임
        config['training']['epochs'] = min(5, config['training']['epochs'])

        # 제안된 변경사항 적용
        for key, value in suggestion['config_changes'].items():
            if key not in config:
                config[key] = {}
            if isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value

        config_name = f"test_config_{i}_{suggestion['name'].replace(' ', '_').lower()}.yaml"
        test_configs.append((suggestion['name'], config, config_name))

    return test_configs


def print_summary(log_data: Dict, suggestions: List[Dict]):
    """결과 요약 출력"""
    print("\n" + "=" * 80)
    print("🎯 RECOMMENDED IMPROVEMENTS")
    print("=" * 80)

    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n{i}. {suggestion['name']}")
        print(f"   📝 {suggestion['description']}")
        print(f"   💡 Reason: {suggestion['reason']}")
        print(f"   ⚙️  Config Changes: {suggestion['config_changes']}")

    print("\n" + "=" * 80)
    print("💻 To test these suggestions:")
    print("=" * 80)
    print("\nExample command:")
    print("  python CR_recon/train.py --config test_config_0_xxx.yaml")
    print("\n비교:")
    print("  1. 첫번째 제안으로 학습 실행 → outputs/train_log.txt 확인")
    print("  2. best_val_loss 값 비교")
    print("  3. 더 낮은 loss를 주는 설정 채택")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Train log 분석 및 하이퍼파라미터 최적화')
    parser.add_argument('--log', required=True, help='Train log 파일 경로')
    parser.add_argument('--base-config', default='CR_recon/configs/default.yaml',
                      help='Base config 파일 경로')
    args = parser.parse_args()

    # Log 파싱
    print(f"\n📖 Parsing log file: {args.log}")
    log_data = parse_train_log(args.log)

    # 제안 생성
    suggestions = generate_suggestions(log_data)

    # 요약 출력
    print_summary(log_data, suggestions)

    # 테스트 config 생성 (선택사항)
    if suggestions:
        print("\n✅ 테스트 config 생성 중...")
        test_configs = create_test_configs(args.base_config, suggestions)
        for name, config, filename in test_configs:
            config_path = Path('CR_recon/configs') / filename
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"  - {filename} (for: {name})")


if __name__ == '__main__':
    main()
