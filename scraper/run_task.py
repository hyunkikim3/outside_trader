from apscheduler.schedulers.blocking import BlockingScheduler

def job_function:
    '''
    define the task you wanna run.

    e.g. print("hello world!")
    '''
    
    return

sched = BlockingScheduler()
# set the parameters as you want
sched.add_job(job_function, 'cron', month='2', day='7-28', hour='9-12', \
              minute='0-59/15')

sched.start()
