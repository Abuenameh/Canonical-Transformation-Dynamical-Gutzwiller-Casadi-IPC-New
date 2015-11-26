/* 
 * File:   main.cpp
 * Author: Abuenameh
 *
 * Created on 17 November 2014, 22:05
 */

#include <cstdlib>
#include <queue>

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/progress.hpp>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/string.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

using namespace boost;
using namespace boost::random;
using namespace boost::filesystem;
using namespace boost::posix_time;
//using namespace boost::interprocess;

using boost::interprocess::shared_memory_object;
using boost::interprocess::managed_shared_memory;
using boost::interprocess::create_only;
using boost::interprocess::open_only;
//using boost::interprocess::allocator;
using boost::interprocess::basic_string;
using boost::interprocess::interprocess_mutex;
using boost::interprocess::interprocess_condition;
//using boost::interprocess::interprocess_;

#include <boost/process.hpp>

using namespace boost::process;
using namespace boost::process::initializers;

#include <nlopt.hpp>

using namespace nlopt;

#include "gutzwiller.hpp"
#include "mathematica.hpp"
#include "casadi.hpp"
#include "orderparameter.hpp"

#include <casadi/interfaces/sundials/cvodes_interface.hpp>

typedef managed_shared_memory::segment_manager segment_manager_t;

typedef interprocess::allocator<void, segment_manager_t> void_allocator;

typedef interprocess::allocator<double, segment_manager_t> double_allocator;
typedef interprocess::vector<double, double_allocator> double_vector;

typedef interprocess::allocator<complex<double>, segment_manager_t> complex_allocator;
typedef interprocess::vector<complex<double>, complex_allocator> complex_vector;
typedef interprocess::allocator<complex_vector, segment_manager_t> complex_vector_allocator;
typedef interprocess::vector<complex_vector, complex_vector_allocator> complex_vector_vector;

typedef interprocess::allocator<char, segment_manager_t> char_allocator;
typedef interprocess::basic_string<char, std::char_traits<char>, char_allocator> char_string;

struct worker_input {
    double Wi;
    double Wf;
    double mu;
    double scale;
    double_vector xi;
    double U0;
    double_vector J0;
    double_vector x0;
    complex_vector_vector f0;
    char_string integrator;

    worker_input(const void_allocator& void_alloc) : xi(void_alloc), J0(void_alloc), x0(void_alloc), f0(void_alloc), integrator(void_alloc) {
    }
};

struct worker_tau {
    interprocess_mutex mutex;
    interprocess_condition cond_empty;
    interprocess_condition cond_full;

    double tau;

    bool full;

    worker_tau() : full(false) {
    }
};

struct worker_output {
    double Ei;
    double Ef;
    double Q;
    double p;
    double_vector Es;
    complex_vector b0;
    complex_vector bf;
    complex_vector_vector f0;
    complex_vector_vector ff;
    char_string runtime;
    bool success;

    interprocess_mutex mutex;
    interprocess_condition cond_empty;
    interprocess_condition cond_full;

    bool full;

    worker_output(const void_allocator& void_alloc) : Es(void_alloc), b0(void_alloc), bf(void_alloc), f0(void_alloc), ff(void_alloc), runtime(void_alloc), full(false) {
    }
};

struct results {
    double tau;
    double Ei;
    double Ef;
    double Q;
    double p;
    double U0;
    vector<double> Es;
    vector<double> J0;
    vector<complex<double>> b0;
    vector<complex<double>> bf;
    vector<vector<complex<double>>> f0;
    vector<vector<complex<double>>> ff;
    string runtime;

//    results(const void_allocator& void_alloc) : Es(void_alloc), J0(void_alloc), b0(void_alloc), bf(void_alloc), f0(void_alloc), ff(void_alloc), runtime(void_alloc) {
//    }
};

double UWi(double W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

struct input {
    double tau;
};

boost::mutex progress_mutex;
boost::mutex inputs_mutex;
boost::mutex problem_mutex;

boost::random::mt19937 rng;
boost::random::uniform_real_distribution<> uni(-1, 1);

void threadfunc(std::string prog, double tauf, queue<input>& inputs, vector<results>& res, progress_display& progress, int thread, managed_shared_memory& segment) {

    void_allocator void_alloc(segment.get_segment_manager());

    std::string tau_name = "tau" + lexical_cast<std::string>(thread);
    worker_tau* tau = segment.construct<worker_tau>(tau_name.c_str())();

    std::string output_name = "output" + lexical_cast<std::string>(thread);
    worker_output* output = segment.construct<worker_output>(output_name.c_str())(void_alloc);

    std::vector<std::string> args;
    args.push_back(prog);
    args.push_back("-1");
    args.push_back(tau_name);
    args.push_back(output_name);

    execute(set_args(args));

    for (;;) {
        input in;
        {
            boost::mutex::scoped_lock lock(inputs_mutex);
            if (inputs.empty()) {
                interprocess::scoped_lock<interprocess_mutex> lock(tau->mutex);

                if (tau->full) {
                    tau->cond_full.wait(lock);
                }
                tau->tau = -1;
                tau->cond_empty.notify_one();
                tau->full = true;
                break;
            }
            in = inputs.front();
            inputs.pop();
        }

        {
            interprocess::scoped_lock<interprocess_mutex> lock(tau->mutex);

            if (tau->full) {
                tau->cond_full.wait(lock);
            }
            tau->tau = in.tau;
            tau->cond_empty.notify_one();
            tau->full = true;
        }

//        results pointRes(void_alloc);
        results pointRes;
        pointRes.tau = in.tau;

        {
            interprocess::scoped_lock<interprocess_mutex> lock(output->mutex);

            if (!output->full) {
                output->cond_empty.wait(lock);
            }
            pointRes.Ei = output->Ei;
            pointRes.Ef = output->Ef;
            pointRes.Q = output->Q;
            pointRes.p = output->p;
            pointRes.Es = vector<double>(output->Es.begin(), output->Es.end());
            pointRes.b0 = vector<complex<double>>(output->b0.begin(), output->b0.end());
            pointRes.bf = vector<complex<double>>(output->bf.begin(), output->bf.end());
            pointRes.runtime = string(output->runtime.begin(), output->runtime.end());
//            pointRes.Es = output->Es;
//            pointRes.b0 = output->b0;
//            pointRes.bf = output->bf;
//            pointRes.f0 = output->f0;
//            pointRes.ff = output->ff;
//            pointRes.runtime = output->runtime;
            output->full = false;
            output->cond_full.notify_one();
        }

        {
            boost::mutex::scoped_lock lock(inputs_mutex);
            res.push_back(pointRes);
        }

        {
            boost::mutex::scoped_lock lock(progress_mutex);
            ++progress;
        }
    }

    segment.destroy_ptr<worker_output>(output);
}

worker_input* initialize(double Wi, double Wf, double mu, double scale, vector<double>& xi, managed_shared_memory& segment) {

    SX f = SX::sym("f", 2 * L * dim);
    SX dU = SX::sym("dU", L);
    SX J = SX::sym("J", L);
    SX U0 = SX::sym("U0");

    U0 = UW(Wi)/scale;
    for (int i = 0; i < L; i++) {
        J[i] = JWij(Wi * xi[i], Wi * xi[mod(i + 1)])/scale;
        dU[i] = UW(Wi * xi[i])/scale - U0;
    }

    SX E = energy(f, J, U0, dU, mu/scale);
    SX E2 = energynew(f, J, U0, dU, mu/scale);
    
    SX g = SX::sym("g", L);
    for (int i = 0; i < L; i++) {
        g[i] = 0;
        for (int n = 0; n < dim; n++) {
            g[i] += f[2*(i*dim+n)]*f[2*(i*dim+n)] + f[2*(i*dim+n)+1]*f[2*(i*dim+n)+1];
        }
    }

    SXFunction nlp("nlp", nlpIn("x", f), nlpOut("f", E));
    NlpSolver solver("solver", "ipopt", nlp, make_dict("hessian_approximation", "limited-memory", "linear_solver", "ma86", "print_level", 0, "print_time", false, "obj_scaling_factor", 1));
//    NlpSolver solver("solver", "ipopt", nlp, make_dict("hessian_approximation", "limited-memory", "linear_solver", "ma86", "print_level", 0, "print_time", false));
    SXFunction nlp2("nlp", nlpIn("x", f), nlpOut("f", E2, "g", g));
    NlpSolver solver2("solver", "ipopt", nlp2, make_dict("hessian_approximation", "limited-memory", "linear_solver", "ma86", "print_level", 0, "print_time", false, "obj_scaling_factor", 1));

    boost::random::mt19937 rng;
    boost::random::uniform_real_distribution<> uni(-1, 1);

    vector<double> xrand(2 * L*dim, 1);
    rng.seed();
    for (int i = 0; i < 2 * L * dim; i++) {
        xrand[i] = (i+1.)/(2*L*dim);//uni(rng);
    }
    vector<double> asd;
    double qwe = nlp2(vector<DMatrix>{xrand, asd})[0].toScalar();
    cout << ::math(qwe) << endl;
    return nullptr;

    map<string, DMatrix> arg;
    arg["lbx"] = -1;
    arg["ubx"] = 1;
    arg["x0"] = xrand;
    arg["lbg"] = 1;
    arg["ubg"] = 1;

    map<string, DMatrix> res = solver2(arg);
    vector<double> x0 = res["x"].nonzeros();
//        vector<double> x0 = xrand;
//    cout << "x0 = " << ::math(x0) << endl;
    cout << "E0 = " << ::math(res["f"].toScalar()) << endl;

    vector<complex<double>> x0i(dim);
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            x0i[n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
        }
        double nrm = sqrt(abs(dot(x0i, x0i)));
        for (int n = 0; n <= nmax; n++) {
            x0[2 * (i * dim + n)] /= nrm;
            x0[2 * (i * dim + n) + 1] /= nrm;
        }
    }
//    cout << "nlp: " << ::math(nlp(vector<DMatrix>{x0, vector<double>()})[0].toScalar()) << endl;

    void_allocator void_alloc(segment.get_segment_manager());
    worker_input* input = segment.construct<worker_input>("input")(void_alloc);
    input->U0 = UW(Wi);
    for (int i = 0; i < L; i++) {
        input->J0.push_back(JWij(Wi * xi[i], Wi * xi[mod(i + 1)]));
    }
    for (int i = 0; i < 2 * L * dim; i++) {
        input->x0.push_back(x0[i]);
    }
    for (int i = 0; i < L; i++) {
        complex_vector f0i(dim, void_alloc);
        for (int n = 0; n <= nmax; n++) {
            f0i[n] = complex<double>(x0[2 * (i * dim + n)], x0[2 * (i * dim + n) + 1]);
        }
        input->f0.push_back(f0i);
    }
    input->Wi = Wi;
    input->Wf = Wf;
    input->mu = mu;
    input->scale = scale;
    for (int i = 0; i < L; i++) {
        input->xi.push_back(xi[i]);
    }

    return input;
}

complex<double> dot(complex_vector& v, complex_vector& w) {
    complex<double> res = 0;
    for (int i = 0; i < v.size(); i++) {
        res += ~v[i] * w[i];
    }
    return res;
}

void evolve(SXFunction& E0, SXFunction& Et, Function& ode_func, vector<double>& p, worker_input* input, worker_output* output, managed_shared_memory& segment) {
    double scale = input->scale;
    double tau = p[L+4]*scale;
//    double tauf = tau;//2e-6;
    double dt = 0.9e-9*scale;
    Integrator integrator_rk("integrator", "rk", ode_func, make_dict("t0", 0, "tf", 2 * tau, "number_of_finite_elements", ceil((2 * tau) / dt)));
    Integrator integrator_cvodes("integrator", "cvodes", ode_func, make_dict("t0", 0, "tf", 2 * tau, "exact_jacobian", false, "max_num_steps", 100000));
    Integrator integrator;
    if (input->integrator == "rk") {
        integrator = integrator_rk;
    }
    if (input->integrator == "cvodes") {
        integrator = integrator_cvodes;
    }

    ptime start_time = microsec_clock::local_time();

    std::vector<double> x0;
    for (int i = 0; i < 2 * L * dim; i++) {
        x0.push_back(input->x0[i]);
    }

    void_allocator void_alloc(segment.get_segment_manager());

//    map<string, DMatrix> res = integrator(make_map("x0", DMatrix(x0), "p", {
//        tau
//    }));
//    map<string, DMatrix> res = integrator(make_map("x0", DMatrix(x0), "p", p));
    integrator.setInput(DMatrix(x0), "x0");
    integrator.setInput(DMatrix(p), "p");
    integrator.reset();
    vector<double> xf;
//    double_vector Es(void_alloc);
    int nt = 50;
    output->Es.clear();
    for (int i = 0; i < nt; i++) {
        integrator.integrate((i+1.)/nt * 2 * tau);
        xf = integrator.getOutput("xf").nonzeros();
//        cout << (i+1.)/nt * 2 * tau << endl;
        output->Es.push_back(Et(vector<DMatrix>{xf, (i+1.)/nt * 2 * tau, tau/scale})[0].toScalar());
    }
//    vector<double> xf = res["xf"].nonzeros();

    complex_vector_vector ff(void_alloc);
    for (int i = 0; i < L; i++) {
        complex_vector ffi(void_alloc);
        for (int n = 0; n <= nmax; n++) {
            ffi.push_back(complex<double>(xf[2 * (i * dim + n)], xf[2 * (i * dim + n) + 1]));
        }
        double nrm = sqrt(abs(dot(ffi, ffi)));
        for (int n = 0; n <= nmax; n++) {
            ffi[n] /= nrm;
        }
        ff.push_back(ffi);
    }
    output->ff = ff;

    complex_vector_vector& f0 = input->f0;

    output->b0.clear();
    output->bf.clear();
    for (int i = 0; i < L; i++) {
        output->b0.push_back(b(f0, i, input->J0, input->U0));
        output->bf.push_back(b(ff, i, input->J0, input->U0));
    }

    output->Ei = E0(vector<DMatrix>{x0})[0].toScalar();
    output->Ef = E0(vector<DMatrix>{xf})[0].toScalar();
    output->Q = output->Ef - output->Ei;

    vector<double> pi(L);
    output->p = 0;
    for (int i = 0; i < L; i++) {
        pi[i] = 1 - norm(dot(ff[i], f0[i]));
        output->p += pi[i];
    }
    output->p /= L;

    ptime stop_time = microsec_clock::local_time();
    time_period period(start_time, stop_time);
    output->runtime = to_simple_string(period.length()).c_str();
}

void fail(std::string error, worker_output* output, managed_shared_memory& segment) {
    void_allocator void_alloc(segment.get_segment_manager());
    output->Ei = numeric_limits<double>::quiet_NaN();
    output->Ef = numeric_limits<double>::quiet_NaN();
    output->Q = numeric_limits<double>::quiet_NaN();
    output->p = numeric_limits<double>::quiet_NaN();
    output->Es = double_vector(1, numeric_limits<double>::quiet_NaN(), void_alloc);
    complex_vector nan_vector(L, numeric_limits<double>::quiet_NaN(), void_alloc);
    output->b0 = nan_vector;
    output->bf = nan_vector;
    complex_vector_vector nan_vector_vector(L, complex_vector(dim, numeric_limits<double>::quiet_NaN(), void_alloc), void_alloc);
    output->f0 = nan_vector_vector;
    output->ff = nan_vector_vector;
    output->runtime = error.c_str();
}

void worker(worker_input* input, worker_tau* tau_in, worker_output* output, managed_shared_memory& segment) {//std::string tau_name, std::string output_name, managed_shared_memory& segment) {

    double Wi = input->Wi;
    double Wf = input->Wf;
    double mu = input->mu;
    double scale = input->scale;
    double_vector xi = input->xi;
    
    vector<double> p(L+5);
    for (int i = 0; i < L; i++) {
        p[i] = xi[i];
    }
    p[L] = Wi;
    p[L+1] = Wf;
    p[L+2] = mu;
    p[L+3] = scale;

    SX f = SX::sym("f", 2 * L * dim);
    SX dU = SX::sym("dU", L);
    SX J = SX::sym("J", L);
    SX U0 = SX::sym("U0");
    SX t = SX::sym("t");

    SX tau = SX::sym("tau");
//    SX tau = p[L+3];

    SX Wt = if_else(t < tau*scale, Wi + (Wf - Wi) * t / (tau*scale), Wf + (Wi - Wf) * (t - tau*scale) / (tau*scale));

    U0 = UW(Wt)/scale;
    for (int i = 0; i < L; i++) {
        J[i] = JWij(Wt * xi[i], Wt * xi[mod(i + 1)])/scale;
        dU[i] = UW(Wt * xi[i])/scale - U0;
    }

    SX E = energy(f, J, U0, dU, mu/scale);
    SXFunction Ef = SXFunction("E",{f, t, tau},
    {
        E
    });
    SXFunction E0 = SXFunction("E0",{f}, Ef(vector<SX>{f, 0, 1}));

//    SX S = canonical(f, J, U0, dU, mu);
//    SXFunction St("St",{t},
//    {
//        S
//    });
//    SX Sdt = St.gradient()(vector<SX>{t})[0];
//
//
//    SXFunction HSr("HSr",{f},
//    {
//        Sdt
//    });
//    SX HSrdf = HSr.gradient()(vector<SX>{f})[0];
//    SXFunction HSi("HSi",{f},
//    {
//        -E
//    });
//    SX HSidf = HSi.gradient()(vector<SX>{f})[0];
//
//    SX ode = SX::sym("ode", 2 * L * dim);
//    for (int j = 0; j < L * dim; j++) {
//        ode[2 * j] = 0;
//        ode[2 * j + 1] = 0;
//        try {
//            ode[2 * j] += 0.5 * HSrdf[2 * j];
//        }
//        catch (CasadiException& e) {
//        }
//        try {
//            ode[2 * j] -= 0.5 * HSidf[2 * j + 1];
//        }
//        catch (CasadiException& e) {
//        }
//        try {
//            ode[2 * j + 1] += 0.5 * HSidf[2 * j];
//        }
//        catch (CasadiException& e) {
//        }
//        try {
//            ode[2 * j + 1] += 0.5 * HSrdf[2 * j + 1];
//        }
//        catch (CasadiException& e) {
//        }
//    }
//    SXFunction ode_func = SXFunction("ode", daeIn("t", t, "x", f, "p", /*tau*/p), daeOut("ode", ode));

    ExternalFunction ode_func("ode");
    
    double taui;
    for (;;) {
        {
            interprocess::scoped_lock<interprocess_mutex> lock(tau_in->mutex);
            if (!tau_in->full) {
                tau_in->cond_empty.wait(lock);
            }
            taui = tau_in->tau;
            tau_in->full = false;
            tau_in->cond_full.notify_one();
            if (taui < 0) {
                break;
            }
        }

        p[L+4] = taui;
        try {
            evolve(E0, Ef, ode_func, p, input, output, segment);
        }
        catch (std::exception& e) {
            fail(e.what(), output, segment);
        }

        {
            interprocess::scoped_lock<interprocess_mutex> lock(output->mutex);

            if (output->full) {
                output->cond_full.wait(lock);
            }
            output->cond_empty.notify_one();
            output->full = true;
        }
    }

    segment.destroy_ptr<worker_tau>(tau_in);
}

void build_ode() {

    SX p = SX::sym("p", L+5);
    SX Wi = p[L];
    SX Wf = p[L+1];
    SX mu = p[L+2];
    SX scale = p[L+3];
    SX tau = p[L+4];

    SX f = SX::sym("f", 2 * L * dim);
    SX dU = SX::sym("dU", L);
    SX J = SX::sym("J", L);
    SX U0 = SX::sym("U0");
    SX t = SX::sym("t");

    SX Wt = if_else(t < tau*scale, Wi + (Wf - Wi) * t / (tau*scale), Wf + (Wi - Wf) * (t - tau*scale) / (tau*scale));

    U0 = UW(Wt)/scale;
    for (int i = 0; i < L; i++) {
        J[i] = JWij(Wt * p[i], Wt * p[mod(i + 1)])/scale;
        dU[i] = UW(Wt * p[i])/scale - U0;
    }

    SX scaledmu = mu/scale;
    SX E = energy(f, J, U0, dU, scaledmu);
    SXFunction Ef = SXFunction("E",{f, t, tau},
    {
        E
    });
    SXFunction E0 = SXFunction("E0",{f}, Ef(vector<SX>{f, 0, 1}));

    SX S = canonical(f, J, U0, dU, scaledmu);
    SXFunction St("St",{t},
    {
        S
    });
    SX Sdt = St.gradient()(vector<SX>{t})[0];


    SXFunction HSr("HSr",{f},
    {
        Sdt
    });
    SX HSrdf = HSr.gradient()(vector<SX>{f})[0];
    SXFunction HSi("HSi",{f},
    {
        -E
    });
    SX HSidf = HSi.gradient()(vector<SX>{f})[0];

    SX ode = SX::sym("ode", 2 * L * dim);
    for (int j = 0; j < L * dim; j++) {
        ode[2 * j] = 0;
        ode[2 * j + 1] = 0;
        try {
            ode[2 * j] += 0.5 * HSrdf[2 * j];
        }
        catch (CasadiException& e) {
        }
        try {
            ode[2 * j] -= 0.5 * HSidf[2 * j + 1];
        }
        catch (CasadiException& e) {
        }
        try {
            ode[2 * j + 1] += 0.5 * HSidf[2 * j];
        }
        catch (CasadiException& e) {
        }
        try {
            ode[2 * j + 1] += 0.5 * HSrdf[2 * j + 1];
        }
        catch (CasadiException& e) {
        }
    }
    SXFunction ode_func = SXFunction("ode", daeIn("t", t, "x", f, "p", p), daeOut("ode", ode));
    
    ode_func.generate("ode");
}

/*
 * 
 */
int main(int argc, char** argv) {
    
//    build_ode();
//    return 0;

    ptime begin = microsec_clock::local_time();

    random::mt19937 rng;
    random::uniform_real_distribution<> uni(-1, 1);

    int seed = lexical_cast<int>(argv[1]);
    
    if (seed != -1) {

        double Wi = lexical_cast<double>(argv[2]);
        double Wf = lexical_cast<double>(argv[3]);

        double mu = lexical_cast<double>(argv[4]);
        
        double scale = lexical_cast<double>(argv[5]);

        double Ui = UWi(Wi);

        double D = lexical_cast<double>(argv[6]);

        double taui = lexical_cast<double>(argv[7]);
        double tauf = lexical_cast<double>(argv[8]);
        int ntaus = lexical_cast<int>(argv[9]);

        int numthreads = lexical_cast<int>(argv[10]);

        int resi = lexical_cast<int>(argv[11]);

        //        int integrator = lexical_cast<int>(argv[11]);
        std::string intg = argv[12];

#ifdef AMAZON
        //    path resdir("/home/ubuntu/Results/Canonical Transformation Dynamical Gutzwiller");
        path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Canonical Transformation Dynamical Gutzwiller 2");
#else
        path resdir("/Users/Abuenameh/Documents/Simulation Results/Canonical Transformation Dynamical Gutzwiller 2");
        //        path resdir("/Users/Abuenameh/Documents/Simulation Results/Dynamical Gutzwiller Hartmann Comparison");
#endif
        if (!exists(resdir)) {
            cerr << "Results directory " << resdir << " does not exist!" << endl;
            exit(1);
        }
        ostringstream oss;
        oss << "res." << resi << ".txt";
        path resfile = resdir / oss.str();
        //#ifndef AMAZON
        while (exists(resfile)) {
            resi++;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }
        //#endif
        if (seed < 0) {
            resi = seed;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }
        vector<double> xi(L, 1);
        rng.seed(seed);
        if (seed > -1) {
            for (int j = 0; j < L; j++) {
                xi[j] = (1 + D * uni(rng));
            }
        }

        //        double Ui = UWi(Wi);
        double mui = mu * Ui;

        filesystem::ofstream os(resfile);
        printMath(os, "int", resi, intg);
        printMath(os, "seed", resi, seed);
        printMath(os, "Delta", resi, D);
        printMath(os, "scale", resi, scale);
        printMath(os, "mures", resi, mui);
        printMath(os, "Ures", resi, Ui);
        printMath(os, "xires", resi, xi);
        os << flush;

        printMath(os, "Wires", resi, Wi);
        printMath(os, "Wfres", resi, Wf);
        os << flush;

        cout << "Res: " << resi << endl;

        struct shm_remove {

            shm_remove() {
                shared_memory_object::remove("SharedMemory");
            }

            ~shm_remove() {
                shared_memory_object::remove("SharedMemory");
            }
        } remover;

        int size = 1000 * (sizeof (worker_input) + numthreads * (sizeof (worker_tau) + sizeof (worker_output))); //2 * (((2 * L * dim + L + 1) + numthreads * (4 * L * dim + 5 * L + 6)) * sizeof (double) +numthreads * 2 * sizeof (ptime)/*sizeof(time_period)*/);

        managed_shared_memory segment(create_only, "SharedMemory", size);

        worker_input* w_input = initialize(Wi, Wf, mui, scale, xi, segment);
        return 0;

        void_allocator void_alloc(segment.get_segment_manager());
        char_string integrator(intg.begin(), intg.end(), void_alloc);
        w_input->integrator = integrator;

        queue<input> inputs;
        if (ntaus == 1) {
            input in;
            in.tau = taui;
            inputs.push(in);
        }
        else {
            for (int i = 0; i < ntaus; i++) {
                input in;
                double tau = taui + i * (tauf - taui) / (ntaus - 1);
                in.tau = tau;
                inputs.push(in);
            }
        }

        vector<results> res;

        progress_display progress(inputs.size());

        thread_group threads;
        for (int i = 0; i < numthreads; i++) {
            threads.create_thread(bind(&threadfunc, argv[0], tauf, boost::ref(inputs), boost::ref(res), boost::ref(progress), i, boost::ref(segment)));
        }
        threads.join_all();

        vector<double> taures;
        vector<double> Eires;
        vector<double> Efres;
        vector<double> Qres;
        vector<double> pres;
        vector<vector<double>> Esres;
        vector<vector<complex<double>>> b0res;
        vector<vector<complex<double>>> bfres;
//        vector<complex_vector_vector> f0res;
//        vector<complex_vector_vector> ffres;
        vector<std::string> runtimeres;

        for (results& ires : res) {
            taures.push_back(ires.tau);
            Eires.push_back(ires.Ei);
            Efres.push_back(ires.Ef);
            Qres.push_back(ires.Q);
            pres.push_back(ires.p);
            Esres.push_back(ires.Es);
            b0res.push_back(ires.b0);
            bfres.push_back(ires.bf);
            runtimeres.push_back(replace_all_copy(ires.runtime, "\"", "\\\""));
//            vector<double> Es(ires.Es.begin(), ires.Es.end());
//            Esres.push_back(Es);
//            vector<complex<double>> b0(ires.b0.begin(), ires.b0.end());
//            vector<complex<double>> bf(ires.bf.begin(), ires.bf.end());
//            b0res.push_back(b0);
//            bfres.push_back(bf);
//            f0res.push_back(ires.f0);
//            ffres.push_back(ires.ff);
//            std::string runtime(ires.runtime.begin(), ires.runtime.end());
//            runtimeres.push_back(replace_all_copy(runtime, "\"", "\\\""));
        }

        printMath(os, "taures", resi, taures);
        printMath(os, "Eires", resi, Eires);
        printMath(os, "Efres", resi, Efres);
        printMath(os, "Qres", resi, Qres);
        printMath(os, "pres", resi, pres);
        printMath(os, "U0res", resi, w_input->U0);
        printMath(os, "J0res", resi, w_input->J0);
        printMath(os, "Esres", resi, Esres);
        printMath(os, "b0res", resi, b0res);
        printMath(os, "bfres", resi, bfres);
        //    printMath(os, "f0res", resi, f0res);
        //    printMath(os, "ffres", resi, ffres);
        printMath(os, "runtime", resi, runtimeres);

        ptime end = microsec_clock::local_time();
        time_period period(begin, end);
        cout << endl << period.length() << endl << endl;

        os << "totalruntime[" << resi << "]=\"" << period.length() << "\";" << endl;

        segment.destroy<worker_input>("input");

    }
    else {
        managed_shared_memory segment(open_only, "SharedMemory");

        worker_input* input = segment.find<worker_input>("input").first;
        worker_tau* tau = segment.find<worker_tau>(argv[2]).first;
        worker_output* output = segment.find<worker_output>(argv[3]).first;

        worker(input, tau, output, segment);

    }

    return 0;

}

