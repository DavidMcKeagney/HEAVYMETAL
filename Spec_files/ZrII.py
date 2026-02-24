#!/usr/bin/python3

from pfac import fac

fac.SetAtom('Zr')

fac.Closed('1s 2s 2p 3s 3p 3d 4s 4p')
fac.Config('4*2 5*1',group='n2')
fac.Config('4*1 5*2',group='n3')
fac.Config('4*3',group='n4')
fac.Config('4*2 6*1',group='n5')
fac.Config('4*1 5*1 6*1',group='n6')
#fac.Config('5*2',group='n4')

fac.ConfigEnergy(0)
fac.ConfigEnergy(1)
fac.ConfigEnergy(2)
fac.ConfigEnergy(3)
fac.ConfigEnergy(4)
fac.OptimizeRadial('n2 n3 n4 n5 n6')
fac.Structure('ZrII.lev.b')
fac.MemENTable('ZrII.lev.b')
fac.PrintTable('ZrII.lev.b', 'ZrII.lev', 1)
