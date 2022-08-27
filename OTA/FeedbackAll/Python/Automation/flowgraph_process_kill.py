import os, signal
  
def process():
     
    # Ask user for the name of process
    name = "flow_graph"
    try:
         
        # iterating through each instance of the process
        for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
            fields = line.split()
             
            # extracting Process ID from the output
            pid = fields[0]
             
            # terminating process
            os.kill(int(pid), signal.SIGKILL)
        print("\nProcess Successfully terminated")
         
    except:
        print("Error Encountered while running script")
  
process()