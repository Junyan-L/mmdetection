from sunpy.time import parse_time
from sunpy.net import hek
from sunpy.net import attrs as a
import pickle as pkl

hek_client = hek.HEKClient()
start_time = parse_time('2014-01-01T00:00:00')
end_time = parse_time('2015-01-01T00:00:00')
response_ch = hek_client.search(a.Time(start_time, end_time),
                                a.hek.CH, a.hek.FRM.Name == 'SPoCA')
# response_fi = hek_client.search(a.Time(start_time, end_time), a.hek.FI)

with open('CH_data_2014.pkl', 'wb') as f:
    pkl.dump(response_ch, f)
# with open('FI_data_2011.pkl', 'wb') as f:
#     pkl.dump(response_fi, f)
