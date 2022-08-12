#include <iostream>
#include <string>
#include<vector>
#include <cmath>
using namespace std;

class american_option
{
public:
    // Class for creating American option objects
    // with following attributes
    double price; // Value of the option
    double Stock; // Current stock price
    double Strike; // Strike price of the option
    american_option(double S, double K, double T, double r, double sigma, int steps);


};

american_option::american_option(double S, double K, double T, double r, double sigma, int steps){
    // This is a constructor for the american_option object
    // which calculates the price of the American option using
    // the binomial model
    Stock=S;
    Strike=K;
    // Initializing a single row
	vector<double> row1(steps+1, 0);
    vector<double> row2(steps+1, 0);
	// Initializing the 2-D vector
    vector<vector<double>> prices(steps+1, row1);
    vector<vector<double>> values(steps+1, row2);
    cout<< values[0][0] << endl;
    double dt=T/steps;
    double disc=exp(-r*dt);
    double a=exp((r+pow(sigma,2))*dt);
    double b=0.5*(disc+a);
    double u=b+sqrt(pow(b,2)-1);
    double d=1/u;
    double p=(exp(r*dt)-d)/(u-d);

    prices[0][0]=S;
    for (int i=1; i<steps+1; i++){
        for (int j=i; j>0; j--){
            prices[j][i]=u*prices[j-1][i-1];
        }
        prices[0][i]=d*prices[0][i-1];

    }
    for (int k=0; k<steps+1; k++ ){
        values[k][steps]=max((K-prices[k][steps]),0.0);
        cout << values[k][steps] <<endl;

    }
    for (int n=steps; n>0; n--){
        for (int i=0; i<steps; i++){
            values[i][n-1]=max((p*values[i+1][n]+(1-p)*(values[i][n]))*disc, max((K-prices[i][n-1]),0.0)); // Using risk neutral approach
        }
    }
    price=values[0][0];

}

int main()
{
    american_option Am(92, 95,1,0.02,0.2,1000); // Example of creating an call object
    cout << "price: "<<Am.price<< endl; // outputting price attribute
    return 0;
}
