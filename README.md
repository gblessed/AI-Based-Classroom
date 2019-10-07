# AI-Based-Classroom
This contains the codes, steps, tests and possible errors for the AI based classroom based on the The Raspberry Pi 3 running with 1GB RAM and Raspbian Stretch lite Operating System (OS).


Steps For Implementation.
	Update and upgrade all packages on the OS. 
	Install python 3.6.0 
	Install the pip tool, which can be used to install required libraries
	Use the pip tool to install the following libraries: nltk (version 3.2.5 is used because newer versions gave errors), numpy, speechrecognition, pyaudio, (reference links in Appendix I) 
	Download the stopwords corpus and WordnetLemmatizer  for the nltk
	Implement the program (Appendix II)
	For an offline test, the audio input should be in wav or flac format
An offline test was carried with the paper.wav file as input.
The Top ten keywords extracted from the abstract of the input  are: network, energy, use, consumption, 2020, learning, case, imt, machine, improve, market, coverage. 

