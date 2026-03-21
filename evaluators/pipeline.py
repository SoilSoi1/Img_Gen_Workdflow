import os
import sys
import torch
from pathlib import Path
from typing import Dict, Optional, List, Union


class EvaluationPipeline:
    """图像生成质量评估统筹管道"""
    
    def __init__(self, device: str = None):
        """
        初始化评估管道
        
        参数:
            device (str): 计算设备，'cuda' 或 'cpu'，None则自动选择
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"评估管道初始化，设备: {self.device}")
    
    def run(self, 
            gen_dir: str,
            real_dir: Optional[str] = None,
            methods: Optional[List[str]] = None,
            lpips_net: str = 'alex',
            lpips_sample_pairs: Optional[int] = None,
            prd_inception_path: Optional[str] = None,
            verbose: bool = True) -> Dict[str, Union[float, dict]]:
        """
        运行评估管道
        
        参数:
            gen_dir (str): 生成图像文件夹路径
            real_dir (str): 真实图像文件夹路径（双数据集方法需要）
            methods (list): 要执行的评估方法列表
                    单数据集方法: ['lpips', 'brisque']
                    双数据集方法: ['fid', 'kid', 'prd']
                    默认: ['fid', 'kid', 'lpips', 'brisque']
            lpips_net (str): LPIPS网络类型 ('alex', 'vgg', 'squeeze')
            lpips_sample_pairs (int): LPIPS的采样对数，None表示计算全部
            prd_inception_path (str): PRD的Inception模型路径
            verbose (bool): 是否打印详细信息
        
        返回:
            dict: 包含所有评估结果的字典
        """
        if methods is None:
            methods = ['fid', 'kid', 'lpips', 'brisque']
        
        results = {}
        
        # 数据集方法分类
        single_dataset_methods = ['lpips', 'brisque']
        double_dataset_methods = ['fid', 'kid', 'prd']
        
        # 检查参数有效性
        for method in methods:
            if method in double_dataset_methods and real_dir is None:
                print(f"警告: {method} 需要real_dir参数，已跳过")
                methods.remove(method)
        
        if verbose:
            print(f"\n执行评估方法: {methods}\n")
        
        # 执行单数据集方法
        if 'lpips' in methods:
            results['lpips'] = self._eval_lpips(gen_dir, net=lpips_net, sample_pairs=lpips_sample_pairs)
        
        if 'brisque' in methods:
            results['brisque'] = self._eval_brisque(gen_dir)
        
        # 执行双数据集方法
        if 'fid' in methods:
            results['fid'] = self._eval_fid(real_dir, gen_dir)
        
        if 'kid' in methods:
            results['kid'] = self._eval_kid(real_dir, gen_dir)
        
        if 'prd' in methods:
            results['prd'] = self._eval_prd(real_dir, gen_dir, inception_path=prd_inception_path)
        
        if verbose:
            self._print_results(results)
        
        return results
    
    def _eval_lpips(self, image_dir: str, net: str = 'alex', sample_pairs: Optional[int] = None) -> float:
        """计算LPIPS"""
        try:
            from lpips_pairwise import cal_lpips_pairwise
            print(f"计算LPIPS (net={net})...")
            score = cal_lpips_pairwise(image_dir, device=self.device, net=net, sample_pairs=sample_pairs)
            return score
        except Exception as e:
            print(f"LPIPS计算失败: {e}")
            return None
    
    def _eval_brisque(self, image_dir: str) -> float:
        """计算BRISQUE"""
        try:
            from brisque_official import cal_brisque_official
            print(f"计算BRISQUE...")
            score = cal_brisque_official(image_dir)
            return score
        except Exception as e:
            print(f"BRISQUE计算失败: {e}")
            return None
    
    def _eval_fid(self, real_dir: str, gen_dir: str) -> float:
        """计算FID"""
        try:
            from _fid import cal_fid
            print(f"计算FID...")
            score = cal_fid(real_dir, gen_dir, device=self.device)
            return score
        except Exception as e:
            print(f"FID计算失败: {e}")
            return None
    
    def _eval_kid(self, real_dir: str, gen_dir: str) -> float:
        """计算KID"""
        try:
            from _kid import cal_kid
            print(f"计算KID...")
            score = cal_kid(real_dir, gen_dir, device=self.device)
            return score
        except Exception as e:
            print(f"KID计算失败: {e}")
            return None
    
    def _eval_prd(self, real_dir: str, gen_dir: str, inception_path: Optional[str] = None) -> dict:
        """计算PRD (Precision & Recall)"""
        try:
            import subprocess
            import json
            import tempfile
            
            print(f"计算PRD...")
            
            prd_script = os.path.join(os.path.dirname(__file__), 'prd', 'prd_from_image_folders.py')
            if not os.path.exists(prd_script):
                raise FileNotFoundError(f"PRD脚本未找到: {prd_script}")
            
            # 构建命令
            cmd = [
                sys.executable,
                prd_script,
                '--reference_dir', real_dir,
                '--eval_dirs', gen_dir,
                '--eval_labels', 'generated',
                '--device', self.device
            ]
            
            if inception_path:
                cmd.extend(['--inception_path', inception_path])
            
            # 执行PRD脚本
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
            
            if result.returncode != 0:
                raise RuntimeError(f"PRD脚本执行失败: {result.stderr}")
            
            # 解析输出获取F-beta分数
            output = result.stdout
            prd_result = self._parse_prd_output(output)
            
            return prd_result
        except Exception as e:
            print(f"PRD计算失败: {e}")
            return None
    
    def _parse_prd_output(self, output: str) -> dict:
        """从PRD输出中提取F-beta分数"""
        result = {}
        try:
            for line in output.split('\n'):
                if 'F-beta' in line or 'Precision' in line or 'Recall' in line:
                    # 尝试解析包含数值的行
                    parts = line.split()
                    for i, part in enumerate(parts):
                        try:
                            value = float(part)
                            if 'F-beta' in line:
                                result['F-beta'] = value
                            elif 'Precision' in line:
                                result['precision'] = value
                            elif 'Recall' in line:
                                result['recall'] = value
                        except ValueError:
                            pass
        except:
            pass
        
        return result if result else {'F-beta': None}
    
    def _print_results(self, results: Dict):
        """打印评估结果"""
        print("\n" + "="*60)
        print("评估结果汇总")
        print("="*60)
        
        for method, score in results.items():
            if score is None:
                print(f"{method:15} : 计算失败")
            elif isinstance(score, dict):
                print(f"{method}:")
                for key, value in score.items():
                    if value is not None:
                        print(f"  {key:15} : {value:.4f}")
                    else:
                        print(f"  {key:15} : N/A")
            else:
                print(f"{method:15} : {score:.4f}")
        
        print("="*60 + "\n")


def main():
    """命令行入口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='图像生成质量评估统筹管道',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--gen_dir', required=True, help='生成图像文件夹路径')
    parser.add_argument('--real_dir', default=None, help='真实图像文件夹路径（若需要FID/KID/PRD）')
    parser.add_argument('--methods', default='fid,kid,lpips,brisque', 
                        help='要执行的评估方法（逗号分隔），可选: fid,kid,lpips,brisque,prd')
    parser.add_argument('--device', default=None, choices=['cuda', 'cpu'], 
                        help='计算设备，默认自动选择')
    parser.add_argument('--lpips_net', default='alex', choices=['alex', 'vgg', 'squeeze'],
                        help='LPIPS网络类型')
    parser.add_argument('--lpips_sample_pairs', type=int, default=None,
                        help='LPIPS采样对数，None表示计算全部')
    parser.add_argument('--prd_inception_path', default=None,
                        help='PRD的Inception模型路径')
    parser.add_argument('--output', default=None, choices=['json', 'dict'],
                        help='输出格式，若指定则保存结果到文件')
    
    args = parser.parse_args()
    
    # 解析methods
    methods = [m.strip() for m in args.methods.split(',')]
    
    # 初始化和运行
    pipeline = EvaluationPipeline(device=args.device)
    results = pipeline.run(
        gen_dir=args.gen_dir,
        real_dir=args.real_dir,
        methods=methods,
        lpips_net=args.lpips_net,
        lpips_sample_pairs=args.lpips_sample_pairs,
        prd_inception_path=args.prd_inception_path
    )
    
    # 保存结果
    if args.output == 'json':
        import json
        output_file = os.path.join(args.gen_dir, 'evaluation_results.json')
        with open(output_file, 'w') as f:
            # 过滤None值
            clean_results = {k: v for k, v in results.items() if v is not None}
            json.dump(clean_results, f, indent=2)
        print(f"结果已保存至: {output_file}")
    
    return results


if __name__ == '__main__':
    main()
