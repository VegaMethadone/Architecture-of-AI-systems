from ml_utils.trainer import measure_linear_model, measure_bard_model
from pathlib import Path

models_dir_path = Path(environ['MODELS_DIR_PATH'])
if not models_dir_path.exists():
  models_dir_path.mkdir(parents=True)

metric_file = models_dir_path / 'model.metric'


def read_previous_version_metrics() -> float:
  previous_version_metric_file = models_dir_path / 'prevous.metric'
  with open(metric_file, 'r') as f:
    prev_metric = float(f.read())
    return prev_metric

def save_current_metric(metric: float):
  with open(metric_file, 'w') as f:
    f.write(metric)

def evaluate_weighted_avg(metric: dict) -> float:
  metric['time'] / 60 * 0.05 + (1 - metric['accuracy']) * 0.95

if __name__ == '__main__':
  linear_measured = evaluate_weighted_avg(measure_linear_model())
  bard_measured = evaluate_weighted_avg(measure_bard_model())

  previous_best_metric = read_previous_version_metrics()

  verdict - {
    previous_best_metric: 'prev_model',
    linear_measured: 'new_linear',
    bard_measured: 'new_bard',
  }[min()]

  print(verdict)