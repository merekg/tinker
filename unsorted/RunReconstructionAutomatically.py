import os
import sys, getopt
import re
import subprocess



#Global Variable declaration for identifying file name

SINGLE = 'Single'
DUAL = 'Dual'


# Global variables for running reconstruction command and acquisition playback commands.
RECONSTRUCTION_APP_COMMAND = ['Reconstruction']
PLAYBACK_APP_PATH = 'AcquisitionPlayback'
PLAYBACK_BASE_FLAGS = ['--verbose=1', '--timeBetweenImagesMs=0'] #It helps pre-corrected images behave like images that need corrections timing wise
PLAYBACK_APP_COMMAND = [PLAYBACK_APP_PATH] + PLAYBACK_BASE_FLAGS



# Utility function to read file names.
def read_files(acquisition_input_directory):
    filenames_list = []
    for files in os.listdir(acquisition_input_directory):
        if files.endswith('.h5'):
            filename, ext = files.split('.h5')
            filenames_list.append(filename)
            
    return filenames_list
        
        
# Function to run Single reconstruction for an acquisition. Which accepts the filename the acq input directory and the recon configFileLocation        
def runSingle(filename,acquisition_input_directory ,configFileLocation):
    print('Running SINGLE for file name ',filename)
    
    #Build a reconstruction command to output a recon with the corresponding acquisition name, the corresponding recon config file and the quit after 1 flag for a single.
    
    # Check if the file requires corrections
    if filename.startswith('IS'):
        reconstructionCommand = RECONSTRUCTION_APP_COMMAND + ['--configuration='+configFileLocation] +['--FileIO.ReconstructedSceneName='+filename, '--quitAfter1', '--FileIO.SaveReconstruction=true']
    else:
        reconstructionCommand = RECONSTRUCTION_APP_COMMAND + ['--configuration='+configFileLocation] +['--FileIO.ReconstructedSceneName='+filename, '--quitAfter1', '--Corrections.ApplyOffsetCorrection=false','--Corrections.ApplyTechniqueCorrection=false',  '--Corrections.ApplySignalInAirCorrection=false',  '--Corrections.ApplyLogging=false',  '--Corrections.ApplyDeadPixelCorrection=false', '--Filters.MedianFilterDiameter=0', '--FileIO.SaveReconstruction=true' ]
        
    
    #Run the command.
    reconProcess = subprocess.Popen(reconstructionCommand, stdout = subprocess.PIPE, stdin = subprocess.PIPE, stderr = subprocess.PIPE)
    
    
    #While the recon command is running start the AcquisitionPlayback command.
    while reconProcess.poll() is None:
        output = reconProcess.stdout.readline()
        milestone1 = output.decode() 
        #Printing Reconstruction output
        print(milestone1)
        
        #Check if the correct file is loaded in the config file.
        if milestone1 == 'Setting configuration: FileIO.ReconstructedSceneName='+filename+'\n':
            
            #Buid and run the playback command for the single.
            acquisitionPath = acquisition_input_directory+filename+'.h5'
            playbackCommand = PLAYBACK_APP_COMMAND + ['-m', acquisitionPath, '--quitAfterPlaylist']
            playbackProcess = subprocess.Popen(playbackCommand, stdout = subprocess.PIPE, stdin = subprocess.PIPE, stderr = subprocess.PIPE)
            
        elif milestone1 == 'Waiting for acquisition!\n':
            
            #Wait for reconstruction to be ready to accept images and send.
            playbackProcess.communicate(input = ('0\n').encode())
    
    
    
    return filename + " done"
            
        
    
    
    
    
# Function to run a Dual reconstruction for a pair of acquisitions. Which accepts the filenames the acq input directory and the recon configFileLocation 
def runDual(filename1, filename2, acquisition_input_directory, configFileLocation):
    
    print('Running DUAL for file name', filename1, filename2)
    
    # Check if the file requires corrections and Build the Reconstruction Command
    if filename1.startswith('IS'):
        reconstructionCommand = RECONSTRUCTION_APP_COMMAND + ['--configuration='+configFileLocation] +['--FileIO.ReconstructedSceneName='+filename2, '--FileIO.SaveReconstruction=false']
    else:
        reconstructionCommand = RECONSTRUCTION_APP_COMMAND + ['--configuration='+configFileLocation] +['--FileIO.ReconstructedSceneName='+filename2, '--Corrections.ApplyOffsetCorrection=false','--Corrections.ApplyTechniqueCorrection=false',  '--Corrections.ApplySignalInAirCorrection=false',  '--Corrections.ApplyLogging=false',  '--Corrections.ApplyDeadPixelCorrection=false', '--Filters.MedianFilterDiameter=0' , '--FileIO.SaveReconstruction=false' ]
    #Run the command.
    reconProcess = subprocess.Popen(reconstructionCommand, stdout = subprocess.PIPE, stdin = subprocess.PIPE, stderr = subprocess.PIPE)
    #Set the stage for a i-
    stage = 0
    while reconProcess.poll() is None:
        output = reconProcess.stdout.readline()
        milestone1 = output.decode() 
        print(milestone1)
        
        ####Check if the correct file is loaded in the config file.
        if milestone1 == 'Setting configuration: FileIO.ReconstructedSceneName='+filename2+'\n' and stage ==0:
            
            ####Buid and run the playback command for the i-
            acquisitionPath1 = acquisition_input_directory+filename1+'.h5'
            playbackCommand = PLAYBACK_APP_COMMAND + ['-m', acquisitionPath1, '--quitAfterPlaylist']
            playbackProcess = subprocess.Popen(playbackCommand, stdout = subprocess.PIPE, stdin = subprocess.PIPE, stderr = subprocess.PIPE)
            playbackProcess.communicate(input = ('0\n').encode())
            #Set the stage for i+
            stage =1
            
        elif milestone1 == 'Waiting for acquisition!\n' and stage ==1:
            
            ####Buid and run the playback command for the i+
            acquisitionPath2 = acquisition_input_directory+filename2+'.h5'
            playbackCommand = PLAYBACK_APP_COMMAND + [ '-p', acquisitionPath2, '--quitAfterPlaylist']
            playbackProcess = subprocess.Popen(playbackCommand, stdout = subprocess.PIPE, stdin = subprocess.PIPE, stderr = subprocess.PIPE)
            playbackProcess.communicate(input = ('0\n').encode())
            #Set the stage for Closing reconstruction
            stage =2
            
            
        elif milestone1 == 'Waiting for acquisition!\n' and stage ==2:
            #Kill Reconstruction
            reconProcess.kill()
    
    return filename2 + ' Done'
    
    


def run(acquisition_input_directory, recon_config_path):
    #read all the acquisitions in and check if they are single or duals then run separate commands
    
    filename_list = read_files(acquisition_input_directory)
    
    for item in filename_list:
        
        if SINGLE in item:
            
            #Call the single reconstruction function
            completion = runSingle(item,acquisition_input_directory ,recon_config_path)
            print(completion)
            print('\n\n\n')
        elif DUAL in item:            
            if 'i-' in item:
                found_dual = False
                first = re.match("(.*?)i\-", item).group()[:-2]
                second = first+'i+'
                for f in filename_list:
                    if second in f:
                        #Call the Dual reconstruction function
                        completion = runDual(item, f,acquisition_input_directory ,recon_config_path)
                        found_dual = True
                        print(completion)
                        print('\n\n\n')
                if not found_dual:
                    print('No corresponding i+ found for'+first+' please refer to the nomenclature rules in README.txt')
            
            


def main(argv):
    acquisition_input_directory = ''
    recon_config_path='/opt/rt3d/etc/reconstruction.conf'
    
    try:
        opts, args = getopt.getopt(argv,"ha:r:",["ackDir=","recon_config_path="])
    except getopt.GetoptError:
        print ('python3 RunReconstructionAutomatically.py -a <ackDir> -r <recon_config_path>')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print ('python3 RunReconstructionAutomatically.py -a or ackDir <ackDir> -r or recon_config_path <recon_config_path>>')
            sys.exit()
        elif opt in ("-a", "--ackDir"):
            acquisition_input_directory = arg
        elif opt in ("-r", "--recon_config_path"):
            recon_config_path = arg
    if acquisition_input_directory=='' or recon_config_path=='':
        print('Please Enter The Corresponding Path for acquisition directory check --help or -h for help ')
    else:
        print('If Recon config path not given the default config path is /opt/rt3d/etc/reconstruction.conf ')
        run(acquisition_input_directory, recon_config_path)


if __name__ == "__main__":
   main(sys.argv[1:])

