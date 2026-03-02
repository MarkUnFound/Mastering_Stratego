/** 
 * File:   scs-greedy.cpp
 * Author: glenn
 *
 * Created on May 13, 2015, 10:52 AM
 */

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <list>

using namespace std;

/** 
 * Greedy SCS
 */
string find_max_overlap(string source, string sink);
void showR(list<string> R){
	for(list<string>::iterator r = R.begin(); r != R.end(); r++ ){
		cout << *r << endl;
	}
}
void greedy_scs(list<string> &R);

// remove substrings
void reduce(list<string> &R){
	list <string> S;
	for (list<string>::iterator x = R.begin(); x != R.end(); x++) {
		bool is_substr = false;
        for (list<string>::iterator y = R.begin(); y != R.end(); y++) {
			if(*x == *y ) continue;
			if((*y).find(*x) != -1){ 
				cout << *x << " " << *y << endl;
				is_substr = true;
			}
		}
		if(!is_substr){
			S.push_back(*x);
		}
	}
	R.clear();
	R.insert(R.end(), S.begin(), S.end());
}

int main(int argc, char** argv) {

    if(argc != 2){
		cout << "Usage: " << argv[0] << " <input-file>" << endl;
		return -1;
	}
	
	int accessions = 50;
	{
    list<string> R;
    string reference;
    string fragment;
    
    string filename = argv[1];
    ifstream fin(filename.c_str());
    if(!fin){
		cout << "Could not open file \'" << filename << "\'" << endl;	
		return -1;
	}
	
    fin >> reference;
    while(fin >> fragment){
		R.push_back(fragment);
    }
    
    //~ R.push_back("AACTG");
    //~ R.push_back("AACA");
    //~ R.push_back("ATC");
    //~ R.push_back("ATCCG");
    //~ R.push_back("AAT");
    //~ R.push_back("AATA");
    //~ R.push_back("AATC");
    //~ R.push_back("AGG");

	//~ reduce(R);
	
	int start_s=clock();
	// the code you wish to time goes here
	greedy_scs(R);
    int stop_s=clock();
	//~ cout << "Time: " << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << endl;
	
    //~ cout << reference << endl;
    //~ cout << reference.length() << endl;
    //~ cout << endl;

    cout << R.front().length() << ", " 
        << (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000 << ", " 
        << R.front() << endl;
}

    return 0;
}

string find_max_overlap(string source, string sink) {

    int k = min(source.length(), sink.length()); // search windows
    string suffix = source.substr(source.length() - 
        k, source.length()); // last k characters
    string prefix = sink.substr(0, k); // first k characters
    string matched = "";
    while (true) {	
        if (suffix == prefix)
            return suffix;
        else {
            k--;
            suffix = source.substr(source.length() - k, source.length());
            prefix = sink.substr(0, k);
        }

    }
    return "";
}

void greedy_scs(list<string> &R) {
    // find the maximum overlap strings

    string overlap_xy, max_overlap_x, max_overlap_y;
    int max_overlap = 0;
    int iter = 0;
    while (R.size() > 1) {

        iter++;
        max_overlap = 0;
        overlap_xy = max_overlap_x = max_overlap_y = "";
		//~ cout << "finding overlap: R = "  << R.size()  << " t = " << iter << endl;
		//~ showR(R);
		
					
        for (list<string>::iterator x = R.begin(); x != R.end(); x++) {
            for (list<string>::iterator y = R.begin(); y != R.end(); y++) {

                if (x == y)
                    continue;

                overlap_xy = find_max_overlap(*x, *y);

                if (overlap_xy.length() >= max_overlap) {
                    max_overlap = overlap_xy.length();
                    max_overlap_x = *(x);
                    max_overlap_y = *(y);

                }
            }
        }

        // cout << "max overlap: " << max_overlap << ": " 
        // << max_overlap_x << ", " << max_overlap_y << endl;

        // remove the two strings
        
        for (list<string>::iterator it = R.begin(); it != R.end(); it++) {
            if (*it == max_overlap_x) {
                it = R.erase(it);
            }
            if (*it == max_overlap_y) {
                it = R.erase(it);
            }   
        }

        // merge
        string merged_xy;
        //~ cout << "maxoverlap : " << max_overlap << endl;
        
		//~ cout << endl;
        //if(max_overlap != 0)
			merged_xy = max_overlap_x.substr(0, max_overlap_x.length() - max_overlap) + max_overlap_y.substr(0, max_overlap_y.length());
		//~ else
			//~ merged_xy = max_overlap_x + max_overlap_y;
			
        if (merged_xy != "")
            R.push_back(merged_xy);


    }

}

