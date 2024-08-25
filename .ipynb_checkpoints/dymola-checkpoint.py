##Date: 07/01/2024    Created by TAMU Dr. O’Neill’s team
from pyfmi import load_fmu


def _main_():
    ## test the 2 fmu files
    #fmu_tes()

    ## run the 2 systems and get results
    Paras=[0.4,0.4,10,0.5,273.15+46,323]
    # number of lists == 30
    # (i=0-5) The first 6 lists: process fan power, regen fan power, DW power, HP power, Heater1 power, Heater2 power
    # (i=6-8),(i=9-11),(i=12-14),(i=15-17),(i=18-20),(i=21-23),(i=24-26),(i=27-29)
    # DWHP1: condenser inlet, condenser outlet, regen inlet, regen outlet(evaporator inlet),
    # evaporator outlet, process inlet, process outlet, food chamber
    # DWHP2: condenser inlet, condenser outlet, regen inlet, regen outlet,
    # evaporator inlet, process inlet(evaporator outlet), process outlet, food chamber
    time1,results1 = tesbed("DWHP1",Paras,600,300)
    time2,results2 = tesbed("DWHP2",Paras,600,300)
    for i,j in zip(results1,results2):
        print(i,j)


def tesbed(system,paras,endtime,timestep):
    ## paras=[fanflow,Tsp_heater1,Tsp_heater2,Toa,Yoa,Speed_HP]
    print(f"UVA_{system}.fmu")
    model=load_fmu(f"UVA_{system}.fmu", kind='cs', log_level=2)
    output=30 ## numbers of outputs
    start_time = 0  #
    end_time = endtime  #
    step_size = timestep  # Define your own step size, unit is second, 300s = 5min

    # Initialize the model with the start time
    model.reset()
    model.initialize(start_time)

    # Create lists to hold the results
    time_results = []  # time index

    # creat empty lists to hold results
    results=[[] for _ in range(output)]

    # Run simulation
    current_time = start_time
    while current_time <= end_time:
        #print(current_time)
        # Set the input values
        #model.set('T_o', paras[0])
        #model.set('W_o', paras[1])
        model.set('FLOW_p', paras[0])
        model.set('FLOW_r', paras[1])
        model.set('N', paras[2])
        model.set('SP_HP', paras[3])
        model.set('Tset1', paras[4])
        model.set('Tset2', paras[5])

        model.do_step(current_time, step_size, True)

        # Get the current values
        time_results.append(current_time)
        for i in range(output):
            results[i].append(model.get(f"y{i}")[0])

        current_time += step_size

    # Terminate the model
    model.terminate()  # This is important, otherwise the FMU will keep running in the background

    # Save the results to a CSV file
    # number of lists==30
    # (i=0-5) The first 6 lists: process fan power, regen fan power, DW power, HP power, Heater1 power, Heater2 power
    # (i=6-8),(i=9-11),(i=12-14),(i=15-17),(i=18-20),(i=21-23),(i=24-26),(i=27-29)
    # DWHP1: condenser inlet, condenser outlet, regen inlet, regen outlet(evaporator inlet),
    # evaporator outlet, process inlet, process outlet, food chamber
    # DWHP2: condenser inlet, condenser outlet, regen inlet, regen outlet,
    # evaporator inlet, process inlet(evaporator outlet), process outlet, food chamber
    return time_results,results


# test the fmu
def fmu_tes():
    paras=[300,0.013,0.4,0.4,10,0.5,323,323]

    time,result = tesbed("DWHP2",paras)
    #time,result = tesbed("DWHP2", paras)
    


_main_()