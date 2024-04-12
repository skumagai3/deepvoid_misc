
#
#
#  Header structure:
#  datatype = 0l
#  gridtype = 0l
#  sizeG    = 0l
#  sizeX    = 0l
#  sizeY    = 0l
#  sizeZ    = 0l
#  offsX    = float(0)
#  offsY    = float(0)
#  offsZ    = float(0)
#  box      = fltarr(3)
#  remaining_bytes = bytarr(128-36-12)
#  header_high     = bytarr(128)

#  Data types:
#  0        bit                 1/8       1                0 -> 1
#  1        signed char          1        8             -128 -> +127
#  2        unsigned char        1        8                0 -> +255
#  3        short int            2       16          -32,768 -> +32,767
#  4        unsigned short int   2       16                0 -> +65,535
#  5        int                  4       32   -2,147,483,648 -> +2,147,483,647
#  6        unsigned int         4       32                0 -> +4,294,967,295
#  7        long int             4       32   -2,147,483,648 -> +2,147,483,647
#  8        float                4       32       3.4 E-38   -> 3.4 E+38
#  9        double               8       64       1.7 E-308  -> 1.7 E+308
# 10        long double          12      96       3.4 E-4932 -> 3.4 E+4932
#
#
#

import struct
import numpy as np
import array as arr
import gzip


#---------------------------------------------------------
#   Lower resolution of cube by half using summation
#---------------------------------------------------------
def shrink(data):
    nx2 = data.shape[0]//2
    ny2 = data.shape[1]//2
    nz2 = data.shape[1]//2
    return data.reshape(nx2,2, ny2,2, nz2,2).sum(axis=1).sum(axis=2).sum(axis=3)
    #return data.reshape(rows, data.shape[0]//rows, cols, data.shape[1]//cols, deps, data.shape[2]//deps).sum(axis=1).sum(axis=2).sum(axis=3)


#---------------------------------------------------------
#
#---------------------------------------------------------
def project_volu(vol, lim1, lim2, ax):
    shp = vol.shape

    if (ax == 0): out = np.zeros(shape=(shp[1], shp[2]))
    if (ax == 1): out = np.zeros(shape=(shp[0], shp[2]))
    if (ax == 2): out = np.zeros(shape=(shp[0], shp[1]))

    if (ax == 0):
        for i in range(lim1,lim2):
            out = out + vol[i,:,:]

    if (ax == 1):
        for i in range(lim1,lim2):
            out = out + vol[:,i,:]

    if (ax == 2):
        for i in range(lim1,lim2):
            out = out + vol[:,:,i]

    return out


#---------------------------------------------------------
#
#---------------------------------------------------------
def rotate_cube(vol, ax, new=False):
    shp = vol.shape

    if (ax == 0):
        for i in range(shp[0]):
            vol[i,:,:] = np.rot90(vol[i,:,:])
    if (ax == 1):
        for i in range(shp[1]):
            vol[:,i,:] = np.rot90(vol[:,i,:])
    if (ax == 2):
        for i in range(shp[2]):
            vol[:,:,i] = np.rot90(vol[:,:,i])
    return vol


#================================================================================
#
#================================================================================


#-----------------------------------
#
#-----------------------------------
def read_volume_header(filename):

    F = open(filename,'rb')
    
    #--- Read header
    head = F.read(256)
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    
    return [sizeX,sizeY,sizeZ]


#-----------------------------------
#
#-----------------------------------
def read_dvolume(filename):

    F = open(filename,'rb')
    
    #--- Read header
    head = F.read(256)
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)

    den = arr.array('d')
    den.fromfile(F,sizeX*sizeY*sizeZ)
    F.close()
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.float64)

    return den
                                                

#-----------------------------------
#
#-----------------------------------
def read_fvolume(filename):
    
    F = open(filename,'rb')

    #--- Read header
    #head = F.read(256)
    head = arr.array('b')
    head.fromfile(F, 256)    
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)
    
    den = arr.array('f')
    den.fromfile(F,sizeX*sizeY*sizeZ)    
    F.close()
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.float32)    
    
    return den

#-----------------------------------
#
#-----------------------------------
def read_ivolume(filename):

    F = open(filename,'rb')

    #--- Read header
    #head = F.read(256)
    head = arr.array('b')
    head.fromfile(F, 256)    
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)
    
    den = arr.array('i')
    den.fromfile(F,sizeX*sizeY*sizeZ)
    F.close()    
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.int32)
    
    return den

#-----------------------------------
# Note: this function read UNSIGNED bytes
#-----------------------------------
def read_bvolume(filename):

    F = open(filename,'rb')
    
    #--- Read header
    head = arr.array('b')
    head.fromfile(F, 256)    
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])    
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)
    
    den = arr.array('B')
    den.fromfile(F,sizeX*sizeY*sizeZ)
    F.close()    
    den = np.array(den).reshape((sizeX,sizeY,sizeZ)).astype(np.uint8)
    
    return den

#-----------------------------------
#
#-----------------------------------
def read_bvolume_gzip(filename):

    with gzip.open(filename, 'rb') as f:
        aaa = np.frombuffer(f.read(), dtype=np.uint8)
    
    (sizeX,) = struct.unpack('i',aaa[12:16])
    (sizeY,) = struct.unpack('i',aaa[16:20])
    (sizeZ,) = struct.unpack('i',aaa[20:24])    
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)

    #--- In this particular case we can use the bytes array
    data = aaa[256:]
    data = np.array(data).reshape((sizeX,sizeY,sizeZ)).astype(np.uint8)
    
    return data

#-----------------------------------
#
#-----------------------------------
def read_ivolume_gzip(filename):
    F = gzip.open(filename, 'rb')
    head = arr.array('b')
    head.fromfile(F, 256)
    (sizeX,) = struct.unpack('i',head[12:16])
    (sizeY,) = struct.unpack('i',head[16:20])
    (sizeZ,) = struct.unpack('i',head[20:24])
    print('>>> Reading volume of size:', sizeX,sizeY,sizeZ)

    data = arr.array('i')
    data.fromfile(F,sizeX*sizeY*sizeZ)
    F.close()
    data = np.array(data).reshape((sizeX,sizeY,sizeZ)).astype(np.int32)
    return data

#-----------------------------------
#
#-----------------------------------
def write_fvolume(vol, filename):

    shp = vol.shape

    #--- Define header
    datatype = 8
    gridtype = 0
    sizeG    = shp[0]
    sizeX    = shp[0]
    sizeY    = shp[1]
    sizeZ    = shp[2]
    offX     = 0.0
    offY     = 0.0
    offZ     = 0.0
    box      = 1.0    
    h0 = np.array([datatype, gridtype, sizeG, sizeX,sizeY,sizeZ], dtype='int32')
    h1 = np.array([offX,offY,offZ], dtype='float32')    
    h2 = np.array([box,box,box], dtype='float32')
    h3 = np.zeros(208,dtype='uint8')

    #--- Binary write
    F = open(filename, "bw")

    #--- Write header to file
    h0.tofile(F)
    h1.tofile(F)
    h2.tofile(F)
    h3.tofile(F)

    #--- write volume data
    vol.astype(dtype='float32').tofile(F)
    
    F.close()

                                                
#-----------------------------------
#
#-----------------------------------
def write_bvolume(vol, filename):
    
    shp = vol.shape
    
    #--- Define header
    datatype = 8
    gridtype = 0
    sizeG    = shp[0]
    sizeX    = shp[0]
    sizeY    = shp[1]
    sizeZ    = shp[2]
    offX     = 0.0
    offY     = 0.0
    offZ     = 0.0
    box      = 1.0
    h0 = np.array([datatype, gridtype, sizeG, sizeX,sizeY,sizeZ], dtype='int32')
    h1 = np.array([offX,offY,offZ], dtype='float32')
    h2 = np.array([box,box,box], dtype='float32')
    h3 = np.zeros(208,dtype='uint8')
    
    #--- Binary write
    F = open(filename, "bw")
    
    #--- Write header to file
    h0.tofile(F)
    h1.tofile(F)
    h2.tofile(F)
    h3.tofile(F)
    
    #--- write volume data
    vol.astype(dtype='uint8').tofile(F)
    
    F.close()


#-----------------------------------
#  https://github.com/SCIInstitute/ImageVis3D/blob/master/doc/import.adoc
#-----------------------------------
def write_raw_volume(_vol, _path,_filename, rescale=False, type='float'):

    if type=='float':
        if rescale == True:
            _vol -= np.min(_vol)
            _vol = _vol/np.max(_vol)
        #--- Binary write
        F = open(_path+_filename + '.raw', "bw")
        _vol.astype(dtype=np.float32).tofile(F)
        F.close()
    if type=='char':
        if rescale == True:
            _vol -= np.min(_vol)
            _vol = _vol/np.max(_vol)*255
        _vol = _vol.astype(np.uint8)

        #--- Binary write
        F = open(_path+_filename + '.raw', "bw")
        _vol.astype(dtype='uint8').tofile(F)
        F.close()
        
    with open(_path + _filename + '.dat', 'w') as f:
        f.write('ObjectFileName: '+ _filename + '.raw' + '\n')
        f.write('TaggedFileName: ---' + '\n')
        f.write('Resolution:     ' + str(_vol.shape[2]) + ' ' + str(_vol.shape[1]) + ' ' + str(_vol.shape[0]) + '\n')
        f.write('SliceThickness: 1 1 1' + '\n')
        
        if type=='float':
            f.write('Format:         FLOAT' + '\n')
        if type=='char':
            f.write('Format:         UCHAR' + '\n')
            
        f.write('NbrTags:        0' + '\n')
        f.write('ObjectType:     TEXTURE_VOLUME_OBJECT' + '\n')
        f.write('ObjectModel:    RGBA' + '\n')
        f.write('GridType:       EQUIDISTANT' + '\n')

def fix_periodic2D(_x,_y, _box):
    
    pos = np.where(_x >= _box)[0]
    if (len(pos) > 0): _x[pos] -= _box
    pos = np.where(_y >= _box)[0]
    if (len(pos) > 0): _y[pos] -= _box

    neg = np.where(_x < 0)[0]
    if (len(neg) > 0): _x[neg] += _box
    neg = np.where(_y < 0)[0]
    if (len(neg) > 0): _y[neg] += _box
    
    pos = np.where(_x >= _box)[0]
    if (len(pos) > 0): _x[pos] -= _box
    pos = np.where(_y >= _box)[0]
    if (len(pos) > 0): _y[pos] -= _box

    neg = np.where(_x < 0)[0]
    if (len(neg) > 0): _x[neg] += _box
    neg = np.where(_y < 0)[0]
    if (len(neg) > 0): _y[neg] += _box

    return _x,_y

#---------------------------------------------------------
#
#---------------------------------------------------------
def read_gadget(file_in):
    file = open(file_in,'rb')
    #--- Read header
    dummy = file.read(4)               
    npart         =  np.fromfile(file, dtype='i', count=6)
    massarr       =  np.fromfile(file, dtype='d', count=6)
    time          = (np.fromfile(file, dtype='d', count=1))[0]
    redshift      = (np.fromfile(file, dtype='d', count=1))[0]
    flag_sfr      = (np.fromfile(file, dtype='i', count=1))[0]
    flag_feedback = (np.fromfile(file, dtype='i', count=1))[0]
    nparttotal    =  np.fromfile(file, dtype='i', count=6)
    flag_cooling  = (np.fromfile(file, dtype='i', count=1))[0]
    NumFiles      = (np.fromfile(file, dtype='i', count=1))[0]
    BoxSize       = (np.fromfile(file, dtype='d', count=1))[0]
    Omega0        = (np.fromfile(file, dtype='d', count=1))[0]
    OmegaLambda   = (np.fromfile(file, dtype='d', count=1))[0]
    HubbleParam   = (np.fromfile(file, dtype='d', count=1))[0]
    header        = file.read(256-6*4 - 6*8 - 8 - 8 - 2*4-6*4 -4 -4 -4*8)
    dummy = file.read(4)
    #--- Particles to read
    n_all = npart[0]+npart[1]+npart[2]+npart[3]+npart[4]
    #--- Read positions
    dummy = file.read(4)
    pos = np.fromfile(file, dtype='f', count=n_all*3)
    file.close()

    #--- Rearrange data
    pos = pos.reshape((n_all,3))
    #--- Only dark matter particles
    x = pos[npart[0]:npart[0]+npart[1],0]
    y = pos[npart[0]:npart[0]+npart[1],1]
    z = pos[npart[0]:npart[0]+npart[1],2]    
    
    return x,y,z, BoxSize