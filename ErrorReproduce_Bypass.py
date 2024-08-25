from pyfmi import load_fmu
import numpy as np
import json
# Define the file path
file_path = 'interruptcontrol.json'

def save_variables(filename, variables):
    with open(filename, 'w') as f:
        json.dump(variables, f)

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)


output=30 ## numbers of outputs
start_time = 0  #
end_time = 6*3600  #
step_size = 300  # Define your own step size, unit is second, 300s = 5min
replaytime = 1
a_old=np.ones([30])


for j in range(replaytime):
    result_all=[]

    print(f'replay time:{j}')
    model=load_fmu("DWHP1_.fmu", kind='cs', log_level=2)
    model.reset()
    model.initialize(start_time)
    print("success boot")

    current_time = start_time

    for i in range(72):
        results=[]
        print(f"current time: {current_time/60} min")
        # paras=data[i]
        paras = [0.4, 0.4, 10, 0.2, 273.15+46, 273.15+46, 0.2]
        model.set('FLOW_p', paras[0])
        model.set('FLOW_r', paras[1])
        model.set('N', paras[2])
        model.set('SP_HP', paras[3])
        model.set('Tset1', paras[4])
        model.set('Tset2', paras[5])
        model.set('Damper', paras[6])

        # print(f"{type()}model.state")

        # print(f"control variables: {paras}")


        model.do_step(current_time, step_size, True)
        for k in range(30):
            # print(k)
            results.append(model.get(f"y{k}")[0])
        print(results)
        result_all.append(results.copy())
        a=np.array(results)
        error_a=a-a_old
        a_old=a
         # print(error_a)

        current_time+=step_size

    model.terminate()

    # print(result_all)
    # print(len(result_all))
    # print(len(data))
    # save_variables('interruptoutput.json', result_all)