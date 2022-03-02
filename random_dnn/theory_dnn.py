"""

"""
from qsub import qsub

mathpath = '~/math12.0/Executables/math'

def submit():
    pbs_array_data = [(alpha100, g100)
                      for alpha100 in range(100, 201, 5)
                      for g100 in range(5, 301, 5)
#                      if alpha100 == 200 or g100 == 300
                      ]
    qsub(f'{mathpath} -script theory_dnn.wls JacobianDOSVectorSave',
         pbs_array_data,
         pass_path=False, path='theory_dnn')



def submit_alpha():
    pbs_array_data = [(alpha100, )
                      for alpha100 in range(100, 201, 5)
                      ]
    qsub(f'{mathpath} -script theory_dnn.wls JacobianOrderedTransitionSave',
         pbs_array_data,
         pass_path=False, path='theory_dnn')




if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python %s FUNCTION_NAME ARG1 ... ARGN' % sys.argv[0])
        quit()
    result = globals()[sys.argv[1]](*sys.argv[2:])
