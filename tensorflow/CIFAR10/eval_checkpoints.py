import argparse
import os
import subprocess
import sys

def main(checkpoints_path, command, start_cnt):
  cnt = start_cnt

  times = {}
  cum_time = 0.0
  with open(os.path.join(checkpoints_path, "times.log"), 'r') as f:
    output = f.read().strip()
    output_lines = output.split('\n')
    for output_line in output_lines:
        [step, time] = output_line.split('\t')
        step = int(step.split(': ')[1])
        time = float(time.split(': ')[1])
        cum_time += time
        times[step] = cum_time

  print("Time (in secs)\tNumber of minibatches\tTop 1 accuracy\tTop 5 accuracy")
  while True:
    ckpt_path = ("%5d" % cnt).replace(' ', '0')
    full_ckpt_path = os.path.join(checkpoints_path, ckpt_path)
    if not os.path.exists(full_ckpt_path):
      break
    if len(os.listdir(full_ckpt_path)) <= 2:
      cnt += 1
      continue
    full_command = command + " --log_root=%s 2>/dev/null" % full_ckpt_path
    output = subprocess.check_output(full_command, shell=True)
    output = output.decode('utf8').strip()
    for line in output.split('\n'):
      if "Precision" in line and "Recall" in line:
        tokens = line.split(", ")  # TODO: Nasty hack, make more robust.
        precision_at_1 = float(tokens[0].split()[-1])
        recall_at_5 = float(tokens[1].split()[-1])
        step = int(tokens[2].split()[3])
        stats = [times[step], step, precision_at_1, recall_at_5]
        print("\t".join([str(stat) for stat in stats]))
        sys.stdout.flush()
    cnt += 1


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description=("Backup model checkpoints periodically")
  )
  parser.add_argument('-i', "--checkpoints_path", type=str, required=True,
                      help="Path to dumped model checkpoints")
  parser.add_argument('-c', "--command", type=str, required=True,
                      help="Command to evaluate each individual checkpoint")
  parser.add_argument('-s', "--start_cnt", type=int, default=1,
                      help="Count to start evaluating checkpoints from")

  cmdline_args = parser.parse_args()
  opt_dict = vars(cmdline_args)

  main(opt_dict["checkpoints_path"], opt_dict["command"], opt_dict["start_cnt"])
