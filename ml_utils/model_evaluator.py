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

if __name__ == '__main__':
  linear_measured = measure_linear_model()
  bard_measured = measure_bard_model()

  # verdict = 'linear' if linear_measured < bard_measured else 'bard'
  if linear_measured < bard_measured:
    save_current_metric(linear_measured)
    verdict = 'linear'
  else:
    save_current_metric(bard_measured)
    verdict = 'bard'

  print(verdict)