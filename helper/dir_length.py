import os
def dir_length(dir):
  initial_count = 0
  for path in os.listdir(dir):
      if os.path.isfile(os.path.join(dir, path)):
          initial_count += 1
  return initial_count