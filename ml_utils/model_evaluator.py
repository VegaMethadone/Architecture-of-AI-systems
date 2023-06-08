from ml_utils.trainer import measure_linear_model, measure_bard_model

def read():
  ...

if __name__ == '__main__':
  linear_measured = measure_linear_model()
  bard_measured = measure_bard_model()

  verdict = 'linear' if linear_measured < bard_measured else 'bard'

  print(verdict)