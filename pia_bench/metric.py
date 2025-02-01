import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from typing import Dict, List
import json
from utils.except_dir import cust_listdir

class MetricsEvaluator:
    def __init__(self, pred_dir: str, label_dir: str, save_dir: str):
        """
        Args:
            pred_dir: 예측 csv 파일들이 있는 디렉토리 경로
            label_dir: 정답 csv 파일들이 있는 디렉토리 경로 
            save_dir: 결과를 저장할 디렉토리 경로
        """
        self.pred_dir = pred_dir
        self.label_dir = label_dir
        self.save_dir = save_dir
        
    def evaluate(self) -> Dict:
        """전체 평가 수행"""
        category_metrics = {}  # 카테고리별 평균 성능 저장
        all_metrics = {  # 모든 카테고리 통합 메트릭
            'falldown': {'f1': [], 'accuracy': [], 'precision': [], 'recall': [], 'specificity': []},
            'violence': {'f1': [], 'accuracy': [], 'precision': [], 'recall': [], 'specificity': []},
            'fire': {'f1': [], 'accuracy': [], 'precision': [], 'recall': [], 'specificity': []}
        }

        # 모든 카테고리의 metrics를 저장할 DataFrame 리스트
        all_categories_metrics = []

        for category in cust_listdir(self.pred_dir):
            if not os.path.isdir(os.path.join(self.pred_dir, category)):
                continue
                
            pred_category_path = os.path.join(self.pred_dir, category)
            label_category_path = os.path.join(self.label_dir, category)
            save_category_path = os.path.join(self.save_dir, category)
            os.makedirs(save_category_path, exist_ok=True)
            
            # 결과 저장을 위한 데이터프레임 생성
            metrics_df = self._evaluate_category(category, pred_category_path, label_category_path)

            metrics_df['category'] = category

            metrics_df.to_csv(os.path.join(save_category_path, f"{category}_metrics.csv"), index=False)
            
            all_categories_metrics.append(metrics_df)

            # 카테고리별 평균 성능 저장
            category_metrics[category] = metrics_df.iloc[-1].to_dict()  # 마지막 row(평균)
            
            # 전체 평균을 위한 메트릭 수집
            # for col in metrics_df.columns:
            #     if col != 'video_name':
            #         event_type, metric_type = col.split('_')
            #         all_metrics[event_type][metric_type].append(category_metrics[category][col])

            for col in metrics_df.columns:
                if col != 'video_name':
                    try:
                        # 첫 번째 언더스코어를 기준으로 이벤트 타입과 메트릭 타입 분리
                        parts = col.split('_', 1)  # maxsplit=1로 첫 번째 언더스코어에서만 분리
                        if len(parts) == 2:
                            event_type, metric_type = parts
                            if event_type in all_metrics and metric_type in all_metrics[event_type]:
                                all_metrics[event_type][metric_type].append(category_metrics[category][col])
                    except Exception as e:
                        print(f"Warning: Could not process column {col}: {str(e)}")
                        continue
        
        # 각 DataFrame에서 마지막 행(average)을 제거
        all_categories_metrics_without_avg = [df.iloc[:-1] for df in all_categories_metrics]
        # 모든 카테고리의 metrics를 하나의 DataFrame으로 합치기
        combined_metrics_df = pd.concat(all_categories_metrics_without_avg, ignore_index=True)
        # 합쳐진 metrics를 json 파일과 같은 위치에 저장
        combined_metrics_df.to_csv(os.path.join(self.save_dir, "all_categories_metrics.csv"), index=False)
        # 결과 출력
        # print("\nCategory-wise Average Metrics:")
        # for category, metrics in category_metrics.items():
        #     print(f"\n{category}:")
        #     for metric_name, value in metrics.items():
        #         if metric_name != "video_name":
        #             print(f"{metric_name}: {value:.3f}")
        
        print("\nCategory-wise Average Metrics:")
        for category, metrics in category_metrics.items():
            print(f"\n{category}:")
            for metric_name, value in metrics.items():
                if metric_name != "video_name":
                    try:
                        if isinstance(value, str):
                            print(f"{metric_name}: {value}")
                        elif metric_name in ['tp', 'tn', 'fp', 'fn']:
                            print(f"{metric_name}: {int(value)}")
                        else:
                            print(f"{metric_name}: {float(value):.3f}")
                    except (ValueError, TypeError):
                        print(f"{metric_name}: {value}")
        # 전체 평균 계산 및 출력
        print("\n" + "="*50)
        print("Overall Average Metrics Across All Categories:")
        print("="*50)
        
        # for event_type in all_metrics:
        #     print(f"\n{event_type}:")
        #     for metric_type, values in all_metrics[event_type].items():
        #         avg_value = np.mean(values)
        #         print(f"{metric_type}: {avg_value:.3f}")
                
        for event_type in all_metrics:
            print(f"\n{event_type}:")
            for metric_type, values in all_metrics[event_type].items():
                avg_value = np.mean(values)
                if metric_type in ['tp', 'tn', 'fp', 'fn']:  # 정수 값
                    print(f"{metric_type}: {int(avg_value)}")
                else:  # 소수점 값
                    print(f"{metric_type}: {avg_value:.3f}")
        ##################################################################################################        
                # 최종 결과를 저장할 딕셔너리
        final_results = {
            "category_metrics": {},
            "overall_metrics": {}
        }
        # 카테고리별 메트릭 저장

        for category, metrics in category_metrics.items():
            final_results["category_metrics"][category] = {}
            for metric_name, value in metrics.items():
                if metric_name != "video_name":
                    if isinstance(value, (int, float)):
                        final_results["category_metrics"][category][metric_name] = float(value)
        
        # 전체 평균 계산 및 저장
        for event_type in all_metrics:
            # print(f"\n{event_type}:")
            final_results["overall_metrics"][event_type] = {}
            for metric_type, values in all_metrics[event_type].items():
                avg_value = float(np.mean(values))
                # print(f"{metric_type}: {avg_value:.3f}")
                final_results["overall_metrics"][event_type][metric_type] = avg_value
        
        # JSON 파일로 저장
        json_path = os.path.join(self.save_dir, "overall_metrics.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, indent=4)

        # return category_metrics
        
        # 누적 메트릭 계산
        accumulated_metrics = self.calculate_accumulated_metrics(combined_metrics_df)
        
        # JSON에 누적 메트릭 추가
        final_results["accumulated_metrics"] = accumulated_metrics
        
        # 누적 메트릭만 따로 저장
        accumulated_json_path = os.path.join(self.save_dir, "accumulated_metrics.json")
        with open(accumulated_json_path, 'w', encoding='utf-8') as f:
            json.dump(accumulated_metrics, f, indent=4)
            
        return accumulated_metrics

    def _evaluate_category(self, category: str, pred_path: str, label_path: str) -> pd.DataFrame:
        """카테고리별 평가 수행"""
        results = []
        metrics_columns = ['video_name']
        
        for pred_file in cust_listdir(pred_path):
            if not pred_file.endswith('.csv'):
                continue
                
            video_name = os.path.splitext(pred_file)[0]
            pred_df = pd.read_csv(os.path.join(pred_path, pred_file))
            
            # 해당 비디오의 정답 CSV 파일 로드
            label_file = f"{video_name}.csv"
            label_path_full = os.path.join(label_path, label_file)
            
            if not os.path.exists(label_path_full):
                print(f"Warning: Label file not found for {video_name}")
                continue
                
            label_df = pd.read_csv(label_path_full)
            
            # 각 카테고리별 메트릭 계산
            video_metrics = {'video_name': video_name}
            categories = [col for col in pred_df.columns if col != 'frame']
            
            for cat in categories:
                # 정답값과 예측값
                y_true = label_df[cat].values
                y_pred = pred_df[cat].values
                
                # 메트릭 계산
                metrics = self._calculate_metrics(y_true, y_pred)
                
                # 결과 저장
                for metric_name, value in metrics.items():
                    col_name = f"{cat}_{metric_name}"
                    video_metrics[col_name] = value
                    if col_name not in metrics_columns:
                        metrics_columns.append(col_name)
            
            results.append(video_metrics)
        
        # 결과를 데이터프레임으로 변환
        metrics_df = pd.DataFrame(results, columns=metrics_columns)
        
        # 평균 계산하여 추가
        avg_metrics = {'video_name': 'average'}
        for col in metrics_columns[1:]:  # video_name 제외
            avg_metrics[col] = metrics_df[col].mean()
            
        metrics_df = pd.concat([metrics_df, pd.DataFrame([avg_metrics])], ignore_index=True)
        
        return metrics_df
    
    # def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    #     """성능 지표 계산"""
    #     tn = np.sum((y_true == 0) & (y_pred == 0))
    #     fp = np.sum((y_true == 0) & (y_pred == 1))
        
    #     metrics = {
    #         'f1': f1_score(y_true, y_pred, zero_division=0),
    #         'accuracy': accuracy_score(y_true, y_pred),
    #         'precision': precision_score(y_true, y_pred, zero_division=0),
    #         'recall': recall_score(y_true, y_pred, zero_division=0),
    #         'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
    #     }
        
    #     return metrics
    

    def calculate_accumulated_metrics(self, all_categories_metrics_df: pd.DataFrame) -> Dict:
        """누적된 혼동행렬로 각 카테고리별 성능 지표 계산"""
        accumulated_results = {"micro_avg": {}}
        categories = ['falldown', 'violence', 'fire']
        
        for category in categories:
            # 해당 카테고리의 혼동행렬 값들 누적
            tp = all_categories_metrics_df[f'{category}_tp'].sum()
            tn = all_categories_metrics_df[f'{category}_tn'].sum()
            fp = all_categories_metrics_df[f'{category}_fp'].sum()
            fn = all_categories_metrics_df[f'{category}_fn'].sum()
            
            # 기본 메트릭 계산
            metrics = {
                'tp': int(tp),
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            }
            
            # 추가 메트릭 계산
            tpr = metrics['recall']  # TPR = recall
            tnr = metrics['specificity']  # TNR = specificity
            
            # Balanced Accuracy
            metrics['balanced_accuracy'] = (tpr + tnr) / 2
            
            # G-Mean
            metrics['g_mean'] = np.sqrt(tpr * tnr) if (tpr * tnr) > 0 else 0
            
            # MCC (Matthews Correlation Coefficient)
            numerator = (tp * tn) - (fp * fn)
            denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            metrics['mcc'] = numerator / denominator if denominator > 0 else 0
            
            # NPV (Negative Predictive Value)
            metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # FAR (False Alarm Rate) = FPR = 1 - specificity
            metrics['far'] = 1 - metrics['specificity']
            
            accumulated_results[category] = metrics
        
        # 전체 카테고리의 누적 값으로 계산
        total_tp = sum(accumulated_results[cat]['tp'] for cat in categories)
        total_tn = sum(accumulated_results[cat]['tn'] for cat in categories)
        total_fp = sum(accumulated_results[cat]['fp'] for cat in categories)
        total_fn = sum(accumulated_results[cat]['fn'] for cat in categories)
        
        # micro average 계산 (전체 누적 값으로 계산)
        accumulated_results["micro_avg"] = {
            'tp': int(total_tp),
            'tn': int(total_tn),
            'fp': int(total_fp),
            'fn': int(total_fn),
            'accuracy': (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn),
            'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
            'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
            'f1': 2 * total_tp / (2 * total_tp + total_fp + total_fn) if (2 * total_tp + total_fp + total_fn) > 0 else 0,
            # ... (다른 메트릭들도 동일한 방식으로 계산)
        }
        
        return accumulated_results
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """성능 지표 계산"""
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        
        metrics = {
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn)
        }
        
        return metrics