#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <algorithm>
using namespace std;

double nCDF(double value)
{
   return 0.5 * erfc(-value /sqrt(2));
}


double black_scholes_call(double S,
                         double r,
                         double sigma,
                         double T,
                         double K){
    double d1=(log(S/K)+(r+pow(sigma,2)*0.5*T))/(sigma*sqrt(T));
    double d2=d1-sigma*sqrt(T);
    double Call=S*nCDF(d1)-K*exp(-r*T)*nCDF(d2);

    return Call;
    }




vector<double> random_walk(const double S,
                           double r,
                           double sigma,
                           double T,
                           int steps){  // This function creates the path of a random walk and stores it in a vector
    default_random_engine generator;
    normal_distribution<double> distribution(0.0,1.0);
    vector<double> walk(steps);
    double dt=T/steps;
    double a=(r-0.5*pow(sigma,2))*dt;
    double b=sigma*sqrt(dt);
    double S_t=S;
    for (int i=0; i<steps; i++){
        S_t=S_t+r*dt+sigma*S_t*sqrt(dt)*distribution(generator); // can also do this by classical way using a and b
        walk[i]=S_t;
    };
    return walk;


    }

vector<double> heston_random_walk(const double S,
                          const double r,
                          const double sigma,
                          const double T,
                          const double rho,
                          const double kappa,
                          const double theta,
                          int steps){
    default_random_engine generator;
    normal_distribution<double> distribution(0.0,1.0);
    vector<double> walk(steps);
    double V_t=pow(sigma,2);
    double dt=T/steps;
    double S_t=S;
    for (int i=0; i<steps; i++){
        double W_1=distribution(generator);
        double W_2=distribution(generator);
        double W_3=rho*W_1 + sqrt(1-pow(rho,2))*W_2; // correlation

        S_t=S_t+(r*S_t*dt+sqrt(V_t)*S_t*W_1);
        V_t=kappa*(theta-V_t)*dt+sqrt(V_t)*W_3;
        walk[i]=S_t;


    }
    return walk;

    }


double european_vanilla(const double S,
                        double K,
                        double r,
                        double sigma,
                        double T,
                        int sim_no){
    // Solving simple geometric Brownian motion
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    double a=(r-0.5*pow(sigma,2))*T;
    double b=sigma*sqrt(T);
    double tot_payoff=0.0;
    for (int i=1;i<=sim_no;i++){
        double S_T=S*exp(a+(b*distribution(generator)));
        tot_payoff+=max(0.0, S_T-K);
    }
    double mean=tot_payoff/double(sim_no);
    return exp(-r*T)*mean;



}

double arithmetic_asian_call(const double S,
                             double K,
                             double r,
                             double sigma,
                             double T,
                             int sim_no,
                             int steps){
    double tot_payoff=0;
    for (int i=0; i<sim_no;i++){
        double tot=0;
        vector<double> prices=random_walk(S,r,sigma, T, steps);
        for (int j=0; j<prices.size(); j++){
            tot+=prices[j];
        }
        double avg=(tot/prices.size()); //prices.size() is just steps
        tot_payoff+=max(0.0,avg-K);

    }
    return exp(-r*T)*(tot_payoff/sim_no);

    }

double out_barrier_call(const double S,
                        const double K,
                        const double r,
                        const double sigma,
                        const double T,
                        const double barrier,
                        const int sim_no,
                        const int steps){
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0,1.0);
        double tot_payoff=0;
        double dt=T/steps;
        double a=(r-0.5*pow(sigma,2))*dt;
        double b=sigma*sqrt(dt);
        for (int i=0; i<sim_no; i++){
            double S_t=S;
            bool above=false;
            for (int j=0; j<steps; j++){
                    S_t=S_t+r*dt+sigma*sqrt(dt)*distribution(generator);
                    if (S_t>barrier){
                        above=true;
                        break;

                    }
                }
            if (above=false){
            tot_payoff+=max(S-K,0.0);
            }

        }
        return exp(-r*T)*(tot_payoff/sim_no);


        }


double barrier_in_call(const double S,
                       double K,
                       double r,
                       double sigma,
                       double T,
                       double barrier,
                       int sim_no,
                       int steps){
        std::default_random_engine generator;
        std::normal_distribution<double> distribution(0.0,1.0);
        double tot_payoff=0;
        double dt=T/steps;
        double a=(r-0.5*pow(sigma,2))*dt;
        double b=sigma*sqrt(dt);
        for (int i=0; i<sim_no; i++){
            double S_t=S;
            for (int j=0; j<steps; j++){
                    S_t=S_t+r*dt+sigma*sqrt(dt)*distribution(generator);
                    if (S_t>barrier){
                        tot_payoff+=black_scholes_call(S_t,r,sigma,(T-(j*dt)),K);
                        break;

                    }
                }


        }
        return exp(-r*T)*(tot_payoff/sim_no);


        }



double lookback_call(const double S,
                     double K,
                     double r,
                     double sigma,
                     double T,
                     int sim_no,
                     int steps){
    double total_payoff=0;
    for (int i=0; i<sim_no; i++){
        vector<double> prices=random_walk(S,r,sigma, T, steps);
        double max_S=*max_element(prices.begin(), prices.end());
        double payoff=max(max_S-K,0.0);
        total_payoff+=payoff;
    }
    return exp(-r*T)*(total_payoff/sim_no);
    }


int main()
{

    cout << "Hello world!" << endl;
    cout << black_scholes_call(100.0, 0.05,0.25, 0.5, 110) << endl;
    cout << european_vanilla(100.0, 110.0, 0.05, 0.25, 0.5, 50000) << endl;
    return 0;
}
