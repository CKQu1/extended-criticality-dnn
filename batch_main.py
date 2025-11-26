import os

def batch_main(arg_strss, script, script_command='python3'): 

    arg_strss = arg_strss.split(';')
    for arg_strs in arg_strss:
        #arg_strs = '--' + 'arg_strs.replace(',',' ')        
        # arg_strs = '--' + ' --'.join(arg_strs.split(','))
        arg_strs = ' '.join(arg_strs.split(','))

        os.system(f'{script_command} {script} {arg_strs}')            

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])