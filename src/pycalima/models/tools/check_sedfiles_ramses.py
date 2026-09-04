import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def read_sed_file(filename,ndust):
    with open(filename, 'r') as f:
        # Read the number of ages and metallicities
        SED_nA, SED_nZ = map(int, f.readline().strip().split())
        
        # Initialize arrays
        SED_ages = np.zeros(SED_nA)
        SED_zeds = np.zeros(SED_nZ)
        SED_table_dust_RAT = np.zeros((SED_nA, SED_nZ, ndust))
        
        # Read data
        for j in range(SED_nZ):
            for i in range(SED_nA):
                for k in range(ndust):
                    line = f.readline().strip()
                    if line:
                        age, zed, sed_value = map(float, line.split())
                        SED_ages[i] = age
                        SED_zeds[j] = zed
                        SED_table_dust_RAT[i, j, k] = sed_value
        
        return SED_ages, SED_zeds, SED_table_dust_RAT

def plot_combined_sed(name,SED_ages, SED_zeds, ndust, combined_sed, photon_groups):
    fig, ax = plt.subplots(1,1, figsize=(5,4),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    ax.set_ylabel(r'SED Value', fontsize=16)
    ax.set_xlabel(r'Photon Group',fontsize=16)
    ax.set_yscale('log')
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")
    colors = ['r','g','b','m','orange','k','c','y']
    for i, age in enumerate(SED_ages):
        for j, zed in enumerate(SED_zeds):
            for k in range(0,ndust):
                ax.plot(photon_groups, combined_sed[i, j, k, :],color=colors[k])
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.2,right=0.98,hspace=0,wspace=0)
    fig.savefig(f'./SEDtable_dust_{name}.png',format='png',dpi=300)

def read_and_plot_mean_cross_sections(dndsize,filenames,labels,title):
    """
    Reads cross-section files and plots them.

    Args:
        filenames (list of str): List of file paths to read.
        labels (list of str): List of labels for each dataset.
        title (str): Title for the plot.
    """
    if len(filenames) != len(labels):
        raise ValueError("Each file must have a corresponding label.")

    fig, ax = plt.subplots(1,1, figsize=(5,4),dpi=300,facecolor='w',edgecolor='k',sharey=True)
    ax.set_ylabel(r'Cross-section [cm$^2$]', fontsize=16)
    ax.set_xlabel(r'Dust temperature [K]',fontsize=16)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.tick_params(labelsize=14)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.minorticks_on()
    ax.tick_params(which='both',axis="both",direction="in")

    for i, filename in enumerate(filenames):
        # Read the file
        data = np.loadtxt(filename)
        
        # First column is Tdust_table, second column is cross-section
        temperatures = data[:, 0]
        cross_sections = data[:, 1]

        # Plot data
        for j in range(0,dndsize):
            ax.plot(temperatures[j*100:(j+1)*100], 
                    cross_sections[j*100:(j+1)*100],
                    label=labels[i]+' '+str(j+1))

    # Customize the plot
    ax.legend(loc='best',fontsize=8,frameon=False)
    fig.subplots_adjust(top=0.99,bottom=0.13,left=0.2,right=0.98,hspace=0,wspace=0)
    fig.savefig(f'./dust_mean_cross_sections.png',format='png',dpi=300)

def main():
    # Get the number of dust components from the user
    ndust = int(input("Enter the number of dust components: "))
    npah = int(input("Enter the number of PAH components: "))
    name = str(input("Enter the name of the dust cross-section type: "))
    
    # Directory containing the SED tables
    directory = './SEDtables/'
    
    # Find all .list files
    file_list = glob.glob(os.path.join(directory, f'SEDtable_dust_{name}*.list'))
    
    # Sort files by photon group number
    file_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0][-1]))
    
    # Read data from the first file to get dimensions
    first_file = file_list[0]
    SED_ages, SED_zeds, _ = read_sed_file(first_file,ndust+2*npah)
    SED_nA = len(SED_ages)
    SED_nZ = len(SED_zeds)
    num_photon_groups = len(file_list)
    
    # Initialize combined SED array
    combined_sed = np.zeros((SED_nA, SED_nZ, ndust+2*npah, num_photon_groups))
    photon_groups = []
    
    for file in file_list:
        # Extract the photon group number from the filename
        ip = int(file.split('_')[-1].split('.')[0][-1])
        photon_groups.append(ip)
        
        # Read the data from the file
        print(file)
        _, _, SED_table_dust = read_sed_file(file,ndust+2*npah)
        combined_sed[:, :, :, ip-1] = SED_table_dust[:,:,:]
    
    # Plot the combined SED
    plot_combined_sed(name,SED_ages, SED_zeds, ndust+2*npah, combined_sed, photon_groups)

    # Read and plot the mean cross-sections
    filenames = [
    './SEDtables/rosseland_mean_graphite.list',
    './SEDtables/rosseland_mean_silicate.list',
    './SEDtables/planck_mean_graphite.list',
    './SEDtables/planck_mean_silicate.list'
    ]
    labels = [
        'Rosseland Mean Graphite',
        'Rosseland Mean Silicate',
        'Planck Mean Graphite',
        'Planck Mean Silicate'
    ]
    read_and_plot_mean_cross_sections(int(ndust/2),filenames,labels,'Dust Mean Cross Sections')

if __name__ == '__main__':
    main()