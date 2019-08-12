import pandas as pd
import numpy as np
import pandas_datareader.data as web
import matplotlib
matplotlib.use('TkAgg') # Remove this to compare with MacOSX backend
import matplotlib.pyplot as plt
import datetime
import scipy.optimize as sco


def extract_data_from_web():
    stocks=['AAPL','WMT','TSLA','GE','AMZN','DB']
    start_date='1/10/2010'
    end_date='01/01/2017'

    def download_data(stocks):
        data = web.DataReader(stocks, data_source='yahoo', start=start_date, end=end_date)['Adj Close']
        data.columns = stocks
        return data


    data=download_data(stocks)

    data.to_csv('data.csv')


if __name__ == '__main__':

    def create_hist_graphs(datachg):
        datachg.hist(bins=100,figsize=(12,6))
        plt.show()


    '''
    #run only when you need to extract data
    extract_data_from_web()
    '''


    def create_multiple_ports(ret,cov,size):
        sim=15000

        all_weights=np.zeros((sim,size))
        ret_arr=np.zeros(sim)
        vol_arr=np.zeros(sim)
        sharpe_arr=np.zeros(sim)

        for x in range(sim):

            weights = np.array(np.random.random(size))
            wt=weights/np.sum(weights)
            all_weights[x,:]=wt

            ret_arr[x]=np.sum(ret*wt)

            vol_arr[x]=np.sqrt(np.dot(wt.T,np.dot(cov,wt)))

            sharpe_arr[x]=ret_arr[x]/vol_arr[x]

        return all_weights,ret_arr,vol_arr,sharpe_arr


    def create_simulated_portfolios(risk,rt,sr,risk_opt,ret_opt):
        plt.scatter(risk, rt,c=sr,cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(risk_opt, ret_opt, c='red', s=50, edgecolors='black')
        plt.show()




    '''read from the csv file'''
    csvdata=pd.read_csv('data.csv')
    dataclean=csvdata.dropna()
    dataclean1=dataclean.set_index('Date')
    datachg=dataclean1.pct_change(1).dropna()

    ret=datachg.mean()*252
    cov=datachg.cov()*252

    create_hist_graphs(datachg)

    '''Create multiple portfolios'''
    wt,rt,risk,sr=create_multiple_ports(ret,cov,len(datachg.columns))

    sharpe_max=sr.max()
    wt_opt=wt[sr.argmax()]
    ret_opt=rt[sr.argmax()]
    risk_opt=risk[sr.argmax()]

    '''Graph simulated portfolios'''
    create_simulated_portfolios(risk,rt,sr,risk_opt,ret_opt)



    ''''############   Run Optimization    ##################'''

    iweights=np.array([.2,.1,.1,.1,.1,.4])

    def get_ret_vol_sr(iweights):
        ret_o=np.sum(iweights*ret)
        vol_o=np.sqrt(np.dot(iweights,np.dot(cov,iweights.T)))
        sr_o=ret_o/vol_o
        return np.array([ret_o,vol_o,sr_o])

    def neg_sharpe(iweights):
        return get_ret_vol_sr(iweights)[2]*-1


    def check_sum(iweights):
        return np.sum(iweights)-1


    cons=({'type':'eq','fun':check_sum})
    bounds=((0,1),(0,1),(0,1),(0,1),(0,1),(0,1))


    # Sequential Least SQuares Programming (SLSQP).
    opt_results=sco.minimize(neg_sharpe,iweights,method='SLSQP',bounds=bounds,constraints=cons)


    opt_sr=opt_results.fun*-1
    opt_wts=opt_results.x*100
    opt_ret=np.sum(opt_wts*ret)
    opt_vol=np.sqrt(np.dot(opt_wts.T,np.dot(cov,opt_wts)))

    df_wts=pd.DataFrame(opt_wts)
    df_wts.columns=['weights']
    df_wts['weights']=df_wts['weights'].apply(lambda x:round(x,2))

    df_names=pd.DataFrame(datachg.columns)
    df_names.columns=['Stocks']

    df_weights=pd.concat([df_names,df_wts],axis=1)



    ''''###########build efficient frontier###########'''

    # Our returns go from 0 to somewhere along 0.3
    # Create a linspace number of points to calculate x on

    frontier_y = np.linspace(0.1,0.35,100)

    def min_vol(iweights):
        return get_ret_vol_sr(iweights)[1]

    frontier_volatility=[]

    for maxreturns in frontier_y:

        cons=({'type':'eq','fun':check_sum},
              {'type':'eq','fun':lambda w: get_ret_vol_sr(w)[0] - maxreturns}
              )

        result=sco.minimize(min_vol,iweights,method='SLSQP',bounds=bounds,constraints=cons)
        frontier_volatility.append(result.fun)


    def create_simulated_portfolios_withEfrontier(risk,rt,risk_opt,ret_opt,frontier_volatility,frontier_y):
        plt.scatter(risk, rt, c=sr, cmap='plasma')
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(risk_opt, ret_opt, c='red', s=50, edgecolors='black')
        plt.plot(frontier_volatility, frontier_y, 'g--', linewidth=3)
        plt.show()


    create_simulated_portfolios_withEfrontier(risk,rt,risk_opt,ret_opt,frontier_volatility,frontier_y)

    print('Simulated max sharpe ratio is {:.2f}'.format(opt_sr))
    print('max weights are')
    print(df_weights)
    print('max return is {}'.format(opt_ret))
    print('min volatility is {}'.format(opt_vol))


