from flask import render_template
from flask import request
from flaskexample import app
import pandas as pd
from flask import session


@app.route('/')
@app.route('/input')
def cesareans_input():
    return render_template("input.html")

@app.route('/Example')
def Example():
  video_url = 'https://www.youtube.com/watch?v=TZeODVlw-_8'
  input2 = video_url
  import webapp_week2
  video_id=webapp_week2.remove_prefix(input2,'https://www.youtube.com/watch?v=')
  video_id=video_id.split('&',1)[0]
  video_link="https://www.youtube.com/embed/{0}".format(video_id)
  some_data=webapp_week2.get_dataset(input2)
  #just select the Cesareans  from the birth dtabase for the month that the user inputs
  pred=webapp_week2.get_results(some_data)
  #pred=pred[0]
  pred_interp='viral, more than 20,000.' if pred>2 else 'low, less than 200.' if pred==0 else 'high, between 2,000 and 19,999.' if pred>1 else 'medium, between 200 and 1,999.'
  #runs python script
  exp1=webapp_week2.get_lime_results(some_data)
  rankingtable=webapp_week2.get_imp_table(some_data,exp1,pred)
  #Want to display youtube video on output page
  #Want to display exp1.show_in_notebook(show_table=True, show_all=False) on output page
  #Create chart from rankingtable and put on output page
  pd.set_option('display.max_colwidth',-1)
  #rec_table=rankingtable.to_html(justify='left',index=False)
  return render_template("output.html", video_link=video_link, result2 = pred_interp, rankingtable=rankingtable)

   
@app.route('/output')
def cesareans_output():
  #pull 'birth_month' from input field and store it
  input2 = request.args.get('video_url')
  import webapp_week2
  video_id=webapp_week2.remove_prefix(input2,'https://www.youtube.com/watch?v=')
  video_id=video_id.split('&',1)[0]
  video_link="https://www.youtube.com/embed/{0}".format(video_id)
  some_data=webapp_week2.get_dataset(input2)
  #just select the Cesareans  from the birth dtabase for the month that the user inputs
  pred=webapp_week2.get_results(some_data)
  #pred=pred[0]
  pred_interp='viral, more than 20,000.' if pred>2 else 'low, less than 200.' if pred==0 else 'high, between 2,000 and 19,999.' if pred>1 else 'medium, between 200 and 1,999.'
  #runs python script
  exp1=webapp_week2.get_lime_results(some_data)
  rankingtable=webapp_week2.get_imp_table(some_data,exp1,pred)
  #Want to display youtube video on output page
  #Want to display exp1.show_in_notebook(show_table=True, show_all=False) on output page
  #Create chart from rankingtable and put on output page
  pd.set_option('display.max_colwidth',-1)
  #rec_table=rankingtable.to_html(justify='left',index=False)
  return render_template("output.html", video_link=video_link, result2 = pred_interp, rankingtable=rankingtable)


