import json
import os
import collections

file_dir = "./_benchmarks_results"

print("summarizing results")

for experiment in os.listdir(file_dir):
  overall_results={}
  summary_path = os.path.join(file_dir,experiment,"metrics.json")
  if not os.path.isfile(summary_path):
    continue
  with open(summary_path,"r") as summary_file:
    print("____________________________________________________")
    print(experiment)
    summary_json = json.load(summary_file)
    completed_results = summary_json["episodes_fully_completed"]

    #objective can be straight, one-turn, navigation, navigation dynamic (Corl)
    # or clear, regular, dense (NoCrash)
    for weather_id,weather in completed_results.items():
      # number_objectives = len(weather)
      weather_results ={}

      num_objective = len(weather)
      for objective_num in range(num_objective):
        results_arr = weather[objective_num]
        weather_results[objective_num] = sum(results_arr) / len(results_arr)
        overall_results[weather_id] = weather_results

  print(overall_results)

