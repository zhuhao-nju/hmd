from __future__ import print_function
import os
import sys
import datetime
import configparser
from LSP import proc_lsp
from LSPET import proc_lspet
from MPII import proc_mpii
from COCO import proc_coco
from H36M import proc_h36m

sys.path.append("../src/")
from utility import take_notes

# parse configures
conf = configparser.ConfigParser()
conf.read(u'../conf.ini', encoding='utf8')
tgt_path = conf.get('DATA', 'tgt_path')
lsp_path = conf.get('DATA', 'lsp_path')
lspet_path = conf.get('DATA', 'lspet_path')
upi_path = conf.get('DATA', 'upi_path')
coco_api_path = conf.get('DATA', 'coco_api_path')
coco_list_path = conf.get('DATA', 'coco_list_path')
h36m_path = conf.get('DATA', 'h36m_path')

c_time = datetime.datetime.now()
time_string = "%s-%02d:%02d:%02d" % (c_time.date(), c_time.hour, c_time.minute, c_time.second)
take_notes("start at %s\n" % time_string, "./data_log.txt", create_file = True)

p_train = 0
p_test = 0

# build all dirs if not exist
for i in [tgt_path + "train/", tgt_path + "train/img/", 
          tgt_path + "train/sil/", tgt_path + "train/para/", 
          tgt_path + "test/", tgt_path + "test/img/", 
          tgt_path + "test/sil/", tgt_path + "test/para/"]:
    if not os.path.exists(i):
        os.makedirs(i)

p_train, p_test = proc_lsp(tgt_path + "train/", tgt_path + "test/",
                                     p_train, p_test,
                                     lsp_path, upi_path)

p_train = proc_lspet(tgt_path + "train/", p_train,
                          lspet_path, upi_path)

p_train, p_test = proc_mpii(tgt_path + "train/", tgt_path + "test/",
                                      p_train, p_test, upi_path)

p_train, p_test = proc_coco(tgt_path + "train/", tgt_path + "test/",
                                      p_train, p_test, coco_list_path)

p_train, p_test = proc_h36m(tgt_path + "train/", tgt_path + "test/",
                                      p_train, p_test, h36m_path)
print("All done")
