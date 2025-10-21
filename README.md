# 481-Project-
DDOS detector with AI integration 

### Load virtual environment 
`venv/scripts/activate` | <= Windows


`source venv/bin/activate` | <= Mac

### Install Req's
`pip install -r requirements.txt`

### CURRENT TODO's:
* ~~Merge all testing and training files into two big dataframes. (This allows the model to generalize and understand different attacks)~~

* Drop Flow ID, Source IP, Destination IP, Source Port, Destination Port features. Not needed.

* Convert to duration-based features instead. (Idk how to do this :sob:)