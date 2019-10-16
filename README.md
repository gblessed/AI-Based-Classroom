# AI-Based-Classroom
This contains the codes, steps, tests and possible errors for the AI based classroom based on the The Raspberry Pi 3 running with 1GB RAM and Raspbian Stretch lite Operating System (OS).


Steps For Implementation.
Update and upgrade all packages on the OS. 
	1. Install python 3.6.0 
      https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tar.xz
   
2. Install the pip tool, which can be used to install required libraries using the command:
        sudo pip3 install --upgrade pip
            

3. Use the pip tool to install the following libraries: nltk (version 3.2.5 is used because newer versions gave errors), numpy, speechrecognition, and pyaudio 

    sudo pip3 install numpy
    
    sudo pip3 install SpeechRecognition
    
    sudo pip3 install nltk==3.2.5
    
    sudo pip3 install python3-pyaudio
    
if you have challenges with the paudio you may have to update the audio libraries :

    sudo apt-get install portaudio19-dev python-all-dev python3-all-dev && sudo pip3 install pyaudio
 
Download the stopwords corpus and WordnetLemmatizer  for the nltk
      

OFFLINE TEST STEPS
The offline test is when you record an audio file and then send to the edge for processing.

Implement the program textrankoffline.py

For an offline test, the audio input should be in wav or flac format

An offline test was carried with the paper.wav file as input.

The Top ten keywords extracted from the abstract of the input  are: network, energy, use, consumption, 2020, learning, case, imt, machine, improve, market, coverage. 


