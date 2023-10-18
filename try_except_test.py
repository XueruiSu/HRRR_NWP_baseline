def command1():  
    assert 1==1
    return 1  
  
def command2():  
    assert 1==1
    return 1  
  
  
def command3():  
    assert 1==1
    return 1  
  
try:  
    a = command1()  
    b = command2()  
    c = command3()  
    all_commands_successful = True  
except Exception as e:  
    print(f"An error occurred: {e}")  
    all_commands_successful = False  
  
if all_commands_successful:  
    print(a, b, c)
    print("All commands were successful, continuing execution...")  
    # 在此处插入后续代码  
else:  
    
    print("One or more commands failed. Stopping execution.")  
