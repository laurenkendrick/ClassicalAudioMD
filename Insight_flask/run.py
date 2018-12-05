#!/usr/bin/env python
from flaskexample import app
from flask import session
#[session.pop(key) for key in list(session.keys())]
app.run(host='0.0.0.0',debug = True)
