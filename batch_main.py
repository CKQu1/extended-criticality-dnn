# import os

# def batch_main(arg_strss, script, script_command='python3', is_args_parser=False): 

#     arg_strss = arg_strss.split(';')
#     for arg_strs in arg_strss:
#         #arg_strs = '--' + 'arg_strs.replace(',',' ')     
#         if is_args_parser:   
#             arg_strs = '--' + ' --'.join(arg_strs.split(','))
#         else:
#             arg_strs = ' '.join(arg_strs.split(','))

#         os.system(f'{script_command} {script} {arg_strs}')            

# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) < 2:
#         print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
#         quit()
#     result = globals()[sys.argv[1]](*sys.argv[2:])

# ---------------------------------------------------------------------------

import argparse
import os

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='nlp-tutorial/main.py training arguments')   
    # training settings 
    #parser.add_argument('--train_with_ddp', default=True, type=bool, help='to use DDP or not')
    parser.add_argument('--arg_strss', type=str)
    parser.add_argument('--script', default='main.py', type=str)
    parser.add_argument('--script_func', type=str)
    parser.add_argument('--script_command', default='python3', type=str)

    args = parser.parse_args()    

    arg_strss = args.arg_strss.split(';')
    for arg_strs in arg_strss:
        #arg_strs = '--' + 'arg_strs.replace(',',' ')        
        # arg_strs = '--' + ' --'.join(arg_strs.split(','))
        arg_strs = ' '.join(arg_strs.split(','))

        os.system(f'{args.script_command} {args.script} {args.script_func} {arg_strs}')  