use bopolyri::{
    mon::Monomial,
    order::{self, DegLex, Lex, MonomialOrdering},
    poly::{self, Polynomial},
    ring::{BoxedRing, Ring},
    var::{AssociatedVariableType, Variable},
};
use core::num;
use fixedbitset::FixedBitSet;
use m4ri_rust::friendly::BinMatrix;
use rand::prelude::SliceRandom;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::{
    alloc::System,
    collections::{BTreeSet, BinaryHeap, HashMap, LinkedList},
    fmt,
    fs::{self, File},
    io::Write,
    marker::PhantomData,
    rc::Rc,
    time::SystemTime,
    u128,
};
use std::{
    cmp::min,
    sync::{Arc, Mutex},
};
use sysinfo::{ProcessExt, SystemExt};

const NR: usize = 10;
const NM: usize = 4096;
const KEY_SIZE: usize = 80;
const NUM_KEYS: usize = 80 + 8 * NR;
const NUM_VARS: usize =
    NUM_KEYS + NM * (LIN_ANALYSIS_LAST_ROUND - LIN_ANALYSIS_START_ROUND + 1) * 32;
const INT_BIT_SIZE: usize = 32;
const ORDERING: DegLex = DegLex;
const LIN_ANALYSIS_START_ROUND: usize = 1;
const LIN_ANALYSIS_LAST_ROUND: usize = NR - 2;
const CHUNK_SIZE: usize = 1024;
const PAR_CHUNK_SIZE: usize = 512;

fn main() {
    // println!("Hello, world!");
    // let ordering = DegLex;
    // let ring = Ring::new(1000, ordering);
    // let v = ring.var(100);
    // let mut vars = HashMap::new();
    // for i in 0..100 {
    //     let name = format!("x_{}", 99 - i);
    //     vars.insert(name.clone(), Variable::new(name, i));
    // }
    // let one = Polynomial::<DegLex>::one();
    // let one2 = Polynomial::one();
    // let one3 = Polynomial::one();
    // let _one4 = Polynomial::<DegLex>::one();
    // let one_m = Monomial::one();
    // let x0 = &one_m * vars.get("x_0").unwrap();
    // let x1x2 = &one_m * vars.get("x_1").unwrap() * vars.get("x_2").unwrap();
    // let _p1 = &one + &one2;
    // let p2 = &one2 + &x0;
    // let p3 = &one3 + x1x2 + &x0 + &one2;
    // println!("Mon: {}", x0);
    // println!("Poly: {}", one);
    // println!("LM Poly: {}", (one + &one2).lm());
    // println!("LM Poly: {}", p2);
    // println!(
    //     "LM Poly: {}",
    //     p3 + Polynomial::from_variable(vars.get("x_44").unwrap())
    // );

    // let x: Vec<_> = (0..4)
    //     .map(|i| Polynomial::from_variable(ring.var(i), ordering))
    //     .collect();
    // let y: Vec<_> = (4..8)
    //     .map(|i| Polynomial::from_variable(ring.var(i), ordering))
    //     .collect();
    // println!("x: {} {} {} {}", x[0], x[1], x[2], x[3]);
    // println!("y: {} {} {} {}", y[0], y[1], y[2], y[3]);
    // let eqs = mibs_box_fwbw(&x, &y);
    // for e in eqs {
    //     println!("{}", e);
    // }

    let ordering = DegLex;
    let mut ring = Ring::new(NUM_VARS);

    eprintln!("Ring size: {}", NUM_VARS);

    for i in 0..NUM_KEYS {
        ring.set_variable_name(format!("K_{}", i), i);
        ring.set_variable_type(AssociatedVariableType::K(i), i);
    }
    let mut pos = NUM_KEYS;
    for i in 0..NM {
        for j in LIN_ANALYSIS_START_ROUND..=LIN_ANALYSIS_LAST_ROUND {
            for k in 0..INT_BIT_SIZE {
                ring.set_variable_name(format!("L_{}_{}_{}", i, j, k), pos);
                ring.set_variable_type(AssociatedVariableType::L(i, j, k), pos);
                pos += 1;
            }
        }
    }
    /*    for i in 0..NM {
        for j in 0..64 {
            ring.set_variable_name(format!("X_{}_{}", i, j), pos);
            ring.set_variable_type(AssociatedVariableType::X(i, j), pos);
            pos += 1;
        }
    }

    for i in 0..NM {
        for j in 0..64 {
            ring.set_variable_name(format!("Y_{}_{}", i, j), pos);
            ring.set_variable_type(AssociatedVariableType::Y(i, j), pos);
            pos += 1;
        }
    }*/
    let ring = &Box::new(ring);

    let mut vars_keys = vec![Polynomial::new(ring); NUM_KEYS];
    let mut vars_l = vec![vec![vec![Monomial::new(ring); 32]; NR]; NM];
    let mut k = vec![Polynomial::new(ring); 80];
    let mut x = vec![vec![Polynomial::<DegLex>::new(ring); 64]; NM];
    let mut y = vec![vec![Polynomial::<DegLex>::new(ring); 64]; NM];
    for i in 0..NUM_KEYS {
        vars_keys[i] = Polynomial::<DegLex>::from_variable(ring, ring.var(i));
    }
    for i in 0..80 {
        k[i] = vars_keys[i].clone();
    }
    let mut pos = NUM_KEYS;
    for i in 0..NM {
        for j in LIN_ANALYSIS_START_ROUND..=LIN_ANALYSIS_LAST_ROUND {
            for k in 0..INT_BIT_SIZE {
                vars_l[i][j][k] = Monomial::from_variable(ring, ring.var(pos));
                pos += 1;
            }
        }
    }
    /*for i in 0..NM {
        for j in 0..64 {
            x[i][j] = Polynomial::from_variable(ring.var(pos));
            pos += 1;
        }
    }

    for i in 0..NM {
        for j in 0..64 {
            y[i][j] = Polynomial::from_variable(ring.var(pos));
            pos += 1;
        }
    }*/
    eprintln!("Generating sample");
    let mut cipher = MibsCipher::new();
    cipher.set_key_tap(&[0x1, 2, 3, 4, 5, 6, 7, 8, 9, 0], &mut [0xFF; NR]);
    let patt = vec![0, 1, 2, 3, 4, 5, 6, 7, 32, 33, 34, 35, 36, 37, 38, 39];
    let list_of_plains: Vec<_> = (0..NM).map(|i| num_to_patt(i, &patt)).collect();
    let list_of_ciphers: Vec<_> = (0..NM)
        .map(|i| {
            let x = (list_of_plains[i].0.as_slice()[0] as u64)
                ^ (list_of_plains[i].0.as_slice()[1] as u64) << 32;
            let mut inter = [0; NR];
            let y = cipher.encrypt_tap(NR, x, &mut inter);
            let mut msg = Message(FixedBitSet::with_capacity(64));
            msg.0.as_mut_slice()[0] = y as u32;
            msg.0.as_mut_slice()[1] = (y >> 32) as u32;
            msg
        })
        .collect();
    for i in 0..NM {
        for j in 0..64 {
            if list_of_plains[i].0[j] {
                x[i][j] = Polynomial::one(ring);
            }
            if list_of_ciphers[i].0[j] {
                y[i][j] = Polynomial::one(ring);
            }
        }
    }

    eprintln!("Round by round forward proning.");
    let s = sysinfo::System::new_all();
    let pid = std::process::id();
    let cur_proc = s.get_process(pid as i32).unwrap();

    let mut excluded_mons = BTreeSet::new();
    //let mut lin_polys = LinkedList::new();

    let num_samples = 2 * CHUNK_SIZE * MibsCipher::intermediate_len();
    //let samples = generate_samples(true, &list_of_plains, true, &mut cipher, num_samples);

    let mut file = File::create("lin_pols.txt").unwrap();

    eprintln!("Small chunk.");
    println!("Cur mem usage: {} kb", cur_proc.memory());
    let (mut polys, num_found_pols) = all_lin_analysis_rbyr_chunk::<DegLex, _>(
        &mut cipher,
        &list_of_plains,
        true,
        &vars_l,
        &excluded_mons,
        true,
        ring,
        CHUNK_SIZE,
    );
    for p in polys.iter() {
        excluded_mons.insert(p.lm().clone());
        //println!("{}", p);
        file.write_all(format!("{}\n", p).as_bytes()).unwrap();
    }
    drop(polys);
    //lin_polys.append(&mut polys);
    println!("Cur mem usage: {} kb", cur_proc.memory());
    println!("#Linear Polynomials: {}", excluded_mons.len());

    eprintln!("Total chunk.");
    println!("Cur mem usage: {} kb", cur_proc.memory());
    let max_req_samples = num_found_pols
        .iter()
        .map(|n| {
            let n = NM * MibsCipher::intermediate_len() - n;
            println!("{}", n);
            n
        })
        .max()
        .unwrap();
    //let num_samples = 5 * max_req_samples / 4;
    //let samples = generate_samples(true, &list_of_plains, true, &mut cipher, num_samples);
    let mut polys = all_lin_analysis_rbyr::<DegLex, _>(
        &mut cipher,
        &list_of_plains,
        true,
        &vars_l,
        &excluded_mons,
        true,
        ring,
    );

     for p in polys.iter() {
        excluded_mons.insert(p.lm().clone());
        file.write_all(format!("{}\n", p).as_bytes()).unwrap();
        //println!("{}", p);
    }
    drop(polys); 
    //lin_polys.append(&mut polys);

    // eprintln!("Round by round backward proning.");
    // let samples = generate_samples(false, &list_of_ciphers, true, &mut cipher);
    // let mut polys =
    //     all_lin_analysis_rbyr::<DegLex, _>(&samples, &vars_l, &excluded_mons, true, ring);
    // for p in polys.iter() {
    //     excluded_mons.insert(p.lm().clone());
    //     println!("{}", p);
    // }
    // lin_polys.append(&mut polys);

    // eprintln!("Total forward proning.");lin_polys
    // let mut polys = all_lin_analysis_tot::<DegLex, _>(
    //     true,
    //     &list_of_plains,
    //     &vars_l,
    //     &excluded_mons,
    //     true,
    //     &mut cipher,
    //     ring,
    // );
    // for p in polys.iter() {
    //     excluded_mons.insert(p.lm().clone());
    //     println!("{}", p);
    // }
    // lin_polys.append(&mut polys);

    // eprintln!("Total backward proning.");
    // let mut polys = all_lin_analysis_tot::<DegLex, _>(
    //     false,
    //     &list_of_ciphers,
    //     &vars_l,
    //     &excluded_mons,
    //     true,
    //     &mut cipher,
    //     ring,
    // );
    // for p in polys.iter() {
    //     excluded_mons.insert(p.lm().clone());
    //     println!("{}", p);
    // }
    // lin_polys.append(&mut polys);

    eprintln!(
        "#Total Vars: {}, #Remove Vars: {}",
        ring.gens(),
        excluded_mons.len()
    );

    //for p in lin_polys.iter() {
    //    file.write_all(format!("{}\n", p).as_bytes()).unwrap();
        //println!("{}", p);
    //}
    //let mut eqs = LinkedList::new();
    //let mut forward_eqs = LinkedList::new();
    // for (i, p) in lin_polys.iter().enumerate() {
    //     if i < 500 {
    //         forward_eqs.push_back(p.clone());
    //     }
    // }

    // aio_encrypt_eqs(&mut eqs, &lin_polys, &k, &x, &y, &vars_keys, &vars_l, ring);
    // for mut p in eqs {
    //     p.justify();
    //     println!("{}", p);
    // }
}

fn num_to_patt(n: usize, patt: &Vec<usize>) -> Message {
    let mut msg = Message(FixedBitSet::with_capacity(64));
    let mut n = n;
    let mut i = 0;
    while n != 0 {
        if n & 0x1 != 0 {
            msg.0.set(patt[i], true);
        }
        n >>= 1;
        i += 1;
    }
    msg
}

fn mibs_box_fwbw<'a, O: MonomialOrdering>(
    x: &[Polynomial<'a, O>],
    y: &[Polynomial<'a, O>],
) -> Vec<Polynomial<'a, O>> {
    let ring = x[0].ring();
    let x = &x.iter().collect::<Vec<_>>();
    let y = &y.iter().collect::<Vec<_>>();
    let mut res = vec![
        x[0] + y[0] * y[1] * y[2]
            + y[0] * y[1]
            + y[0] * y[3]
            + y[0]
            + y[1] * y[2] * y[3]
            + y[1] * y[3]
            + y[1]
            + y[2]
            + Monomial::one(ring),
        x[1] + y[0] * y[1] * y[3]
            + y[0] * y[1]
            + y[0] * y[2] * y[3]
            + y[1] * y[2]
            + y[1]
            + y[2] * y[3]
            + y[2]
            + Monomial::one(ring),
        x[2] + y[0] * y[1] * y[3]
            + y[0] * y[1]
            + y[0] * y[2] * y[3]
            + y[0] * y[3]
            + y[1] * y[2] * y[3]
            + y[1] * y[2]
            + y[1] * y[3]
            + y[2]
            + y[3]
            + Monomial::one(ring),
        x[3] + y[0] * y[1] * y[2]
            + y[0] * y[2] * y[3]
            + y[0]
            + y[1] * y[2] * y[3]
            + y[1] * y[3]
            + y[1],
        y[0] + x[0] * x[1] * x[2]
            + x[0] * x[1] * x[3]
            + x[0] * x[3]
            + x[0]
            + x[1] * x[2] * x[3]
            + x[1] * x[3]
            + x[1]
            + x[2]
            + x[3],
        y[1] + x[0] * x[1] * x[2]
            + x[0] * x[1] * x[3]
            + x[0] * x[2] * x[3]
            + x[0]
            + x[1] * x[2]
            + x[1] * x[3]
            + x[1]
            + x[3],
        y[2] + x[0] * x[1] * x[3]
            + x[0] * x[2] * x[3]
            + x[0] * x[2]
            + x[0] * x[3]
            + x[1] * x[2]
            + x[1]
            + x[3]
            + Monomial::one(ring),
        y[3] + x[0] * x[1] * x[2]
            + x[0] * x[2]
            + x[0]
            + x[1] * x[2] * x[3]
            + x[1] * x[3]
            + x[2]
            + x[3],
    ];
    for p in res.iter_mut() {
        p.justify();
    }
    res
}

fn mibs_lin<'a, O: MonomialOrdering>(x: &[Polynomial<'a, O>]) -> Vec<Polynomial<'a, O>> {
    let ring = x[0].ring();
    let x = &x.iter().collect::<Vec<_>>();
    let mut t = vec![Polynomial::zero(ring); 32];
    for i in 0..4 {
        t[7 * 4 + i] =
            x[6 * 4 + i] + x[5 * 4 + i] + x[4 * 4 + i] + x[3 * 4 + i] + x[2 * 4 + i] + x[1 * 4 + i];
        t[6 * 4 + i] =
            x[7 * 4 + i] + x[5 * 4 + i] + x[4 * 4 + i] + x[2 * 4 + i] + x[1 * 4 + i] + x[0 * 4 + i];
        t[5 * 4 + i] =
            x[7 * 4 + i] + x[6 * 4 + i] + x[4 * 4 + i] + x[3 * 4 + i] + x[1 * 4 + i] + x[0 * 4 + i];
        t[4 * 4 + i] =
            x[7 * 4 + i] + x[6 * 4 + i] + x[5 * 4 + i] + x[3 * 4 + i] + x[2 * 4 + i] + x[0 * 4 + i];
        t[3 * 4 + i] = x[7 * 4 + i] + x[6 * 4 + i] + x[4 * 4 + i] + x[3 * 4 + i] + x[2 * 4 + i];
        t[2 * 4 + i] = x[7 * 4 + i] + x[6 * 4 + i] + x[5 * 4 + i] + x[2 * 4 + i] + x[1 * 4 + i];
        t[1 * 4 + i] = x[6 * 4 + i] + x[5 * 4 + i] + x[4 * 4 + i] + x[1 * 4 + i] + x[0 * 4 + i];
        t[0 * 4 + i] = x[7 * 4 + i] + x[5 * 4 + i] + x[4 * 4 + i] + x[3 * 4 + i] + x[0 * 4 + i];
    }
    t
}

fn mibs_inv_lin<'a, O: MonomialOrdering>(x: &[Polynomial<'a, O>]) -> Vec<Polynomial<'a, O>> {
    let ring = x[0].ring();
    let x = &x.iter().collect::<Vec<_>>();
    let mut t = vec![Polynomial::zero(ring); 32];
    for i in 0..4 {
        t[7 * 4 + i] = x[7 * 4 + i] + x[6 * 4 + i] + x[3 * 4 + i] + x[2 * 4 + i] + x[1 * 4 + i];
        t[6 * 4 + i] = x[6 * 4 + i] + x[5 * 4 + i] + x[2 * 4 + i] + x[1 * 4 + i] + x[0 * 4 + i];
        t[5 * 4 + i] = x[5 * 4 + i] + x[4 * 4 + i] + x[3 * 4 + i] + x[1 * 4 + i] + x[0 * 4 + i];
        t[4 * 4 + i] = x[7 * 4 + i] + x[4 * 4 + i] + x[3 * 4 + i] + x[2 * 4 + i] + x[0 * 4 + i];
        t[3 * 4 + i] =
            x[7 * 4 + i] + x[6 * 4 + i] + x[4 * 4 + i] + x[2 * 4 + i] + x[1 * 4 + i] + x[0 * 4 + i];
        t[2 * 4 + i] =
            x[7 * 4 + i] + x[6 * 4 + i] + x[5 * 4 + i] + x[3 * 4 + i] + x[1 * 4 + i] + x[0 * 4 + i];
        t[1 * 4 + i] =
            x[6 * 4 + i] + x[5 * 4 + i] + x[4 * 4 + i] + x[3 * 4 + i] + x[2 * 4 + i] + x[0 * 4 + i];
        t[0 * 4 + i] =
            x[7 * 4 + i] + x[5 * 4 + i] + x[4 * 4 + i] + x[3 * 4 + i] + x[2 * 4 + i] + x[1 * 4 + i];
    }
    t
}

fn mibs_perm<'a, O: MonomialOrdering>(x: &[Polynomial<'a, O>]) -> Vec<Polynomial<'a, O>> {
    let ring = x[0].ring();
    let x = &x.iter().collect::<Vec<_>>();
    let mut t = vec![Polynomial::zero(ring); 32];
    for i in 0..4 {
        t[6 * 4 + i] = x[7 * 4 + i].clone();
        t[0 * 4 + i] = x[6 * 4 + i].clone();
        t[7 * 4 + i] = x[5 * 4 + i].clone();
        t[5 * 4 + i] = x[4 * 4 + i].clone();
        t[2 * 4 + i] = x[3 * 4 + i].clone();
        t[1 * 4 + i] = x[2 * 4 + i].clone();
        t[4 * 4 + i] = x[1 * 4 + i].clone();
        t[3 * 4 + i] = x[0 * 4 + i].clone();
    }
    t
}

fn mibs_inv_perm<'a, O: MonomialOrdering>(x: &[Polynomial<'a, O>]) -> Vec<Polynomial<'a, O>> {
    let ring = x[0].ring();
    let x = &x.iter().collect::<Vec<_>>();
    let mut t = vec![Polynomial::zero(ring); 32];
    for i in 0..4 {
        t[7 * 4 + i] = x[6 * 4 + i].clone();
        t[6 * 4 + i] = x[0 * 4 + i].clone();
        t[5 * 4 + i] = x[7 * 4 + i].clone();
        t[4 * 4 + i] = x[5 * 4 + i].clone();
        t[3 * 4 + i] = x[2 * 4 + i].clone();
        t[2 * 4 + i] = x[1 * 4 + i].clone();
        t[1 * 4 + i] = x[4 * 4 + i].clone();
        t[0 * 4 + i] = x[3 * 4 + i].clone();
    }
    t
}

fn mibs_key_round_eqs<'a, O: MonomialOrdering>(
    keqs: &mut LinkedList<Polynomial<'a, O>>,
    rnd: usize,
    stt: &mut Vec<Polynomial<'a, O>>,
    vars_key: &[Polynomial<'a, O>],
) {
    let mut t = vec![Polynomial::zero(stt[0].ring()); 80];
    for i in 0..80 {
        t[i] = stt[(i + 19) % 80].clone();
    }
    let ee = mibs_box_fwbw(&t[76..80], &vars_key[80 + rnd * 8 + 4..80 + rnd * 8 + 8]);
    for e in ee {
        keqs.push_back(e);
    }
    let ee = mibs_box_fwbw(&t[72..76], &vars_key[80 + rnd * 8 + 0..80 + rnd * 8 + 4]);
    for e in ee {
        keqs.push_back(e);
    }
    for i in 0..8 {
        t[72 + i] = vars_key[80 + rnd * 8 + i].clone();
    }

    for i in 0..5 {
        t[14 + i] += (((rnd + 1) >> i) & 0x1) as u64;
    }
    *stt = t;
}

fn aio_encrypt_eqs<'a, O: MonomialOrdering>(
    eqs: &mut LinkedList<Polynomial<'a, O>>,
    forward_eqs: &LinkedList<Polynomial<'a, O>>,
    k: &[Polynomial<'a, O>],
    x: &[Vec<Polynomial<'a, O>>],
    y: &[Vec<Polynomial<'a, O>>],
    vars_key: &[Polynomial<'a, O>],
    vars_l: &[Vec<Vec<Monomial<'a, O>>>],
    ring: &'a BoxedRing<O>,
) {
    let mut stt = k.to_vec();
    let mut l = vec![vec![vec![Polynomial::new(ring); 32]; NR + 2]; NM];
    for i in 0..NM {
        for j in 0..32 {
            l[i][0][j] = x[i][j].clone();
            l[i][1][j] = x[i][32 + j].clone();
            l[i][NR][j] = y[i][32 + j].clone();
            l[i][NR + 1][j] = y[i][j].clone();
        }
        for j in LIN_ANALYSIS_START_ROUND..=LIN_ANALYSIS_LAST_ROUND {
            for k in 0..32 {
                l[i][j + 1][k] = Polynomial::from_monomial(ring, vars_l[i][j][k].clone());
            }
        }
    }

    for p in forward_eqs {
        let m = p.lm();
        if m.degree() == 1 {
            let vars = m.vars().unwrap();
            if let AssociatedVariableType::L(i, j, k) = ring.var(vars[0].order()).associated_type()
            {
                //eprintln!("-L: {}", l[i][j + 1][k]);
                //eprintln!("P: {}", p);
                l[i][j + 1][k] += p;
                //eprintln!("+L: {}", l[i][j + 1][k]);
            }
        }
    }

    for i in 0..NR {
        let mut rk = vec![Polynomial::new(ring); 32];
        mibs_key_round_eqs(eqs, i, &mut stt, &vars_key);
        rk[..32].clone_from_slice(&stt[48..(32 + 48)]);
        for l_j in l.iter() {
            let mut tt = rk.clone();
            let mut t = l_j[i].clone();
            for k in 0..32 {
                tt[k] += &l_j[i + 1][k];
                t[k] += &l_j[i + 2][k];
            }
            let t = mibs_inv_perm(&t);
            let t = mibs_inv_lin(&t);
            for k in 0..8 {
                let ee = mibs_box_fwbw(&tt[k * 4..(k + 1) * 4], &t[k * 4..(k + 1) * 4]);
                for e in ee {
                    if !e.is_zero() {
                        eqs.push_back(e);
                    }
                }
                //eprintln!("- {}, {}, {} -", i, j, k);
            }
        }
    }
}

const MIBS_SBOX: [u8; 16] = [4, 15, 3, 8, 13, 10, 12, 0, 11, 5, 7, 14, 2, 6, 1, 9];
#[derive(Clone)]
struct MibsCipher {
    round_keys: [u32; NR],
}

impl MibsCipher {
    fn new() -> MibsCipher {
        MibsCipher {
            round_keys: [0u32; NR],
        }
    }

    fn set_key_tap(&mut self, main_key: &[u8], intermediate: &mut [u8]) {
        fn rotr_80_19(n: &mut u128) {
            *n = (*n >> 19) ^ (*n << (80 - 19));
            *n &= 0xFFFFFFFFFFFFFFFFFFFF;
        }
        let mut key: u128 = 0;
        for i in 0..10 {
            key <<= 8;
            key ^= main_key[9 - i] as u128;
        }
        for r in 0..NR {
            //eprintln!("{:#X}", key);
            rotr_80_19(&mut key);
            //eprintln!("{:#X}", key);
            let sbox = (MIBS_SBOX[(key >> (80 - 4)) as usize] as u8) << 4
                | (MIBS_SBOX[(key >> (80 - 8)) as usize & 0xF] as u8);
            let tmp = (r + 1) & 0x1F;
            key = (key & 0x00FFFFFFFFFFFFFFFFFF) | (sbox as u128) << (80 - 8);
            key ^= (tmp as u128) << 14;
            self.round_keys[r] = (key >> (80 - 32)) as u32;
            intermediate[r] = (self.round_keys[r] >> (32 - 8)) as u8;
        }
    }

    #[inline]
    fn m1(a: u32, r1: usize, r2: usize, r3: usize, r4: usize, r5: usize, r6: usize) -> u32 {
        (a << (r1 * 4))
            ^ (a << (r2 * 4))
            ^ (a << (r3 * 4))
            ^ (a << (r4 * 4))
            ^ (a << (r5 * 4))
            ^ (a << (r6 * 4))
    }

    #[inline]
    fn m2(a: u32, r1: usize, r2: usize, r3: usize, r4: usize, r5: usize) -> u32 {
        (a << (r1 * 4)) ^ (a << (r2 * 4)) ^ (a << (r3 * 4)) ^ (a << (r4 * 4)) ^ (a << (r5 * 4))
    }

    #[inline]
    fn permute(a: u32) -> u32 {
        let mut res: u32;
        res = Self::m1(a, 0, 1, 3, 4, 6, 7) & 0xF0000000;
        res |= (Self::m1(a, 1, 2, 3, 4, 5, 6) >> 4) & 0x0F000000;
        res |= (Self::m1(a, 0, 1, 2, 4, 5, 7) >> 8) & 0x00F00000;
        res |= (Self::m2(a, 1, 2, 3, 6, 7) >> 12) & 0x000F0000;
        res |= (Self::m2(a, 0, 2, 3, 4, 7) >> 16) & 0x0000F000;
        res |= (Self::m2(a, 0, 1, 3, 4, 5) >> 20) & 0x00000F00;
        res |= (Self::m2(a, 0, 1, 2, 5, 6) >> 24) & 0x000000F0;
        res |= (Self::m1(a, 0, 2, 3, 5, 6, 7) >> 28) & 0x000000F;
        res
    }

    #[inline]
    fn round_f(left: u32, round_key: u32, tap: &mut u32) -> u32 {
        *tap = left;
        let mut xor_data = left ^ round_key;

        let mut out = 0;
        for i in 0..8 {
            let tmp = xor_data & 0xF;
            out ^= (MIBS_SBOX[tmp as usize] as u32) << 28;
            xor_data >>= 4;
            if i != 7 {
                out >>= 4
            }
        }
        Self::permute(out)
    }

    fn encrypt_tap(&self, nr: usize, plain: u64, intermediate: &mut [u32]) -> u64 {
        let mut left = (plain >> 32) as u32;
        let mut right = plain as u32;
        for r in 0..nr {
            //eprintln!("r {} l: {:#X}", r, left);
            //eprintln!("r {} k: {:#X}", r, self.round_keys[r]);
            let f = Self::round_f(left, self.round_keys[r], &mut intermediate[r]);
            let t = f ^ right;
            right = left;
            left = t;
        }
        let mut cipher = right as u64;
        cipher <<= 32;
        cipher ^= left as u64;
        cipher
    }

    fn decrypt_tap(&self, nr: usize, cipher: u64, intermediate: &mut [u32]) -> u64 {
        let mut right = (cipher >> 32) as u32;
        let mut left = cipher as u32;
        for r in (0..nr).rev() {
            let f = Self::round_f(left, self.round_keys[r], &mut intermediate[r]);
            let t = f ^ right;
            right = left;
            left = t;
        }
        let mut plain = left as u64;
        plain <<= 32;
        plain ^= right as u64;
        plain
    }
}

#[derive(Clone)]
struct Message(FixedBitSet);

#[derive(Clone)]
struct IntermediateData(FixedBitSet);

#[derive(Clone)]
struct Key(FixedBitSet);

trait Cipher: Sized {
    fn new() -> Self;
    fn set_random_key(&mut self) -> Key;
    fn set_zero_key(&mut self) -> Key;
    fn generate_forward_sample(&self, zz: &[Message], sample: &mut Sample<Self>);
    fn generate_backward_sample(&self, zz: &[Message], sample: &mut Sample<Self>);
    fn block_len() -> usize;
    fn key_len() -> usize;
    fn intermediate_len() -> usize;
}

struct Sample<C> {
    key: Key,
    x: Vec<Message>,
    y: Vec<Message>,
    d: Vec<Vec<IntermediateData>>,
    _marker: PhantomData<C>,
}

impl<C: Cipher> Sample<C> {
    fn new(nm: usize) -> Self {
        Sample {
            key: Key(FixedBitSet::with_capacity(NUM_KEYS)),
            x: vec![Message(FixedBitSet::with_capacity(C::block_len())); nm],
            y: vec![Message(FixedBitSet::with_capacity(C::block_len())); nm],
            d: vec![
                vec![IntermediateData(FixedBitSet::with_capacity(C::intermediate_len())); NR];
                nm
            ],
            _marker: PhantomData,
        }
    }

    fn apply_monomial<'a, O: MonomialOrdering>(&self, mon: &Monomial<O>) -> bool {
        let mut ans = true;
        if mon.is_zero() {
            return false;
        }
        if mon.is_one() {
            return true;
        }
        let ring = mon.ring().unwrap();
        if let Some(vars) = mon.vars() {
            let iter = vars.iter().map(|v| ring.var(v.order()));
            for v in iter {
                match v.associated_type() {
                    AssociatedVariableType::K(i) => {
                        ans &= self.key.0[i];
                    }
                    AssociatedVariableType::X(i, j) => {
                        ans &= self.x[i].0[j];
                    }
                    AssociatedVariableType::Y(i, j) => {
                        ans &= self.y[i].0[j];
                    }
                    AssociatedVariableType::L(i, j, k) => {
                        ans &= self.d[i][j].0[k];
                    }
                    _ => panic!("Invalid Mon : {}", v),
                }
            }
        }
        ans
    }
}

impl Sample<MibsCipher> {}
impl Cipher for MibsCipher {
    fn new() -> Self {
        MibsCipher::new()
    }
    fn block_len() -> usize {
        64
    }
    fn key_len() -> usize {
        80
    }
    fn intermediate_len() -> usize {
        32
    }
    fn set_zero_key(&mut self) -> Key {
        let mut key = Key(FixedBitSet::with_capacity(NUM_KEYS));
        let mut inter = [0u8; NR];
        let mut key_u8 = [0; 10];
        for i in 0..KEY_SIZE {
            if key.0[i] {
                key_u8[i / 8] ^= 0x1 << (i % 8);
            }
        }
        self.set_key_tap(&key_u8, &mut inter);
        for (i, d) in inter.iter().enumerate() {
            for j in 0..8 {
                key.0.set(KEY_SIZE + i * 8 + j, (d & (0x1 << j)) != 0);
            }
        }
        key
    }

    fn set_random_key(&mut self) -> Key {
        let mut key = Key(FixedBitSet::with_capacity(NUM_KEYS));
        for i in 0..KEY_SIZE {
            key.0.set(i, rand::random());
        }
        let mut inter = [0u8; NR];
        let mut key_u8 = [0; 10];
        for i in 0..KEY_SIZE {
            if key.0[i] {
                key_u8[i / 8] ^= 0x1 << (i % 8);
            }
        }
        self.set_key_tap(&key_u8, &mut inter);
        for (i, d) in inter.iter().enumerate() {
            for j in 0..8 {
                key.0.set(KEY_SIZE + i * 8 + j, (d & (0x1 << j)) != 0);
            }
        }
        key
    }

    fn generate_forward_sample(&self, zz: &[Message], sample: &mut Sample<MibsCipher>) {
        sample
            .x
            .resize_with(zz.len(), || Message(FixedBitSet::with_capacity(64)));
        sample.x.clone_from_slice(zz);
        for (i, msg) in zz.iter().enumerate() {
            let mut inter = [0u32; NR];
            let x = (msg.0.as_slice()[0] as u64) ^ (msg.0.as_slice()[1] as u64) << 32;
            self.encrypt_tap(NR, x, &mut inter);
            for (j, d) in inter.iter().enumerate() {
                sample.d[i][j].0.as_mut_slice()[0] = *d;
            }
        }
    }

    fn generate_backward_sample(&self, zz: &[Message], sample: &mut Sample<MibsCipher>) {
        sample.y.clone_from_slice(zz);
        for (i, msg) in zz.iter().enumerate() {
            let mut inter = [0u32; NR];
            let y = (msg.0.as_slice()[1] as u64) ^ (msg.0.as_slice()[0] as u64) << 32;
            self.decrypt_tap(NR, y, &mut inter);
            for (j, d) in inter.iter().enumerate() {
                sample.d[i][j].0.as_mut_slice()[0] = *d;
            }
        }
    }
}

const EXECIVE_SAMPLES: usize = 256;
fn generate_samples<C: Cipher>(
    is_forward: bool,
    list_of_messages: &[Message],
    print: bool,
    cipher: &mut C,
    num_samples: usize,
) -> Vec<Sample<C>> {
    let zz = list_of_messages;
    //let num_samples = 2 * NM * C::intermediate_len();
    let mut samples = Vec::with_capacity(num_samples);
    samples.resize_with(num_samples, || {
        let mut smp = Sample::<C>::new(NM);
        smp.key = cipher.set_random_key();
        if is_forward {
            cipher.generate_forward_sample(&zz, &mut smp);
        } else {
            cipher.generate_backward_sample(&zz, &mut smp);
        }
        smp
    });
    samples
}

fn prone_polynomials<'a, O: MonomialOrdering, C: Cipher + Sync>(
    list_of_monos: &[Monomial<'a, O>],
    is_forward: bool,
    cipher: &mut C,
    zz: &[Message],
    print: bool,
    ring: &'a BoxedRing<O>,
) -> LinkedList<Polynomial<'a, O>> {
    let dim_samples = list_of_monos.len();
    let num_samples =
        ((dim_samples + EXECIVE_SAMPLES + PAR_CHUNK_SIZE - 1) / PAR_CHUNK_SIZE) * PAR_CHUNK_SIZE;
    let mut arc_mat = Arc::new(Mutex::new(BinMatrix::zero(
        dim_samples,
        dim_samples + num_samples,
    )));
    if print {
        eprintln!(
            "Matrix size: {} x {}",
            dim_samples,
            dim_samples + num_samples
        );
    }

    (0..num_samples / PAR_CHUNK_SIZE)
        .into_par_iter()
        .for_each(|ii| {
            let arc_mat = arc_mat.clone();
            let mut cipher = C::new();
            let mut smp = Sample::<C>::new(NM);
            for iii in 0..PAR_CHUNK_SIZE {
                let i = ii * PAR_CHUNK_SIZE + iii;
                smp.key = cipher.set_random_key();
                if is_forward {
                    cipher.generate_forward_sample(&zz, &mut smp);
                } else {
                    cipher.generate_backward_sample(&zz, &mut smp);
                }
                let mut mat = arc_mat.lock().unwrap();
                for (j, m) in list_of_monos.iter().enumerate() {
                    let v = smp.apply_monomial(m);
                    mat.set_bit(j, i, v);
                }
            }
        });
    let mut mat = arc_mat.lock().unwrap();
    let mat_time = SystemTime::now();

    for i in 0..dim_samples {
        mat.set_bit(i, num_samples + i, true);
    }
    if print {
        eprintln!("Echelonizing matrix")
    }
    mat.echelonize_full();
    let wnd = mat.get_window(0, 0, dim_samples, num_samples);
    let rank = wnd.rank();
    if print {
        eprintln!("Matrix Rank: {}", rank);
    }
    if print {
        eprintln!(
            "Matrix Time: {} s",
            mat_time.elapsed().unwrap().as_secs_f32()
        );
    }
    let mut res = LinkedList::new();
    for i in rank..dim_samples {
        let mut poly = Polynomial::<O>::zero(ring);
        for j in num_samples..num_samples + dim_samples {
            if mat.bit(i, j) {
                poly += &list_of_monos[j - num_samples];
            }
        }
        if !poly.is_zero() {
            res.push_back(poly);
        }
    }
    res
}

fn all_lin_analysis_rbyr_chunk<'a, O: MonomialOrdering, C: Cipher + Clone + Sync>(
    cipher: &mut C,
    zz: &[Message],
    is_forward: bool,
    vars_l: &Vec<Vec<Vec<Monomial<'a, O>>>>,
    excluded_mons: &BTreeSet<Monomial<'a, O>>,
    print: bool,
    ring: &'a BoxedRing<O>,
    chunk_size: usize,
) -> (LinkedList<Polynomial<'a, O>>, Vec<usize>) {
    let tot_time = SystemTime::now();
    let mut vars = Vec::new();
    let mut eqs = LinkedList::new();
    let num_chunks = (NM + chunk_size - 1) / chunk_size;
    let mut num_found_pols = Vec::new();
    num_found_pols.resize(NR, 0);
    for i in LIN_ANALYSIS_START_ROUND..=LIN_ANALYSIS_LAST_ROUND {
        for kk in 0..num_chunks {
            let time = SystemTime::now();
            vars.clear();
            for k in (kk * chunk_size)..min((kk + 1) * chunk_size, NM) {
                for j in 0..C::intermediate_len() {
                    if !excluded_mons.contains(&vars_l[k][i][j]) {
                        vars.push(vars_l[k][i][j].clone());
                    }
                }
            }
            vars.push(Monomial::one(ring));
            vars.sort();
            vars.reverse();

            let mut teqs = prone_polynomials::<O, C>(&vars, is_forward, cipher, zz, print, ring);
            let new_pols = teqs.len();
            for p in teqs {
                //excluded_mons.insert(p.lm().clone());
                eqs.push_back(p);
            }
            num_found_pols[i] += new_pols;

            if print {
                eprintln!("---------");
                eprintln!(
                    "Rnd: {}, #eqs: {} , duration: {} s",
                    i,
                    new_pols,
                    time.elapsed().unwrap().as_secs_f32()
                );
                eprintln!("---------");
            }
        }
    }
    if print {
        eprintln!("---------");
        eprintln!(
            "#Tot eqs: {} , duration: {} ",
            eqs.len(),
            tot_time.elapsed().unwrap().as_secs_f32()
        );
        eprintln!("---------");
    }

    (eqs, num_found_pols)
}

fn all_lin_analysis_rbyr<'a, O: MonomialOrdering, C: Cipher + Clone + Sync>(
    cipher: &mut C,
    zz: &[Message],
    is_forward: bool,
    vars_l: &Vec<Vec<Vec<Monomial<'a, O>>>>,
    excluded_mons: &BTreeSet<Monomial<'a, O>>,
    print: bool,
    ring: &'a BoxedRing<O>,
) -> LinkedList<Polynomial<'a, O>> {
    let tot_time = SystemTime::now();
    let mut vars = Vec::new();
    let mut eqs = LinkedList::new();
    for i in LIN_ANALYSIS_START_ROUND..=LIN_ANALYSIS_LAST_ROUND {
        let time = SystemTime::now();
        vars.clear();
        for k in 0..NM {
            for j in 0..C::intermediate_len() {
                if !excluded_mons.contains(&vars_l[k][i][j]) {
                    vars.push(vars_l[k][i][j].clone());
                }
            }
        }
        vars.push(Monomial::one(ring));
        vars.sort();
        vars.reverse();

        let mut teqs = prone_polynomials::<O, C>(&vars, is_forward, cipher, zz, print, ring);
        let new_pols = teqs.len();
        for p in teqs {
            //excluded_mons.insert(p.lm().clone());
            eqs.push_back(p);
        }

        if print {
            eprintln!("---------");
            eprintln!(
                "Rnd: {}, #eqs: {} , duration: {} s",
                i,
                new_pols,
                time.elapsed().unwrap().as_secs_f32()
            );
            eprintln!("---------");
        }
    }
    if print {
        eprintln!("---------");
        eprintln!(
            "#Tot eqs: {} , duration: {} ",
            eqs.len(),
            tot_time.elapsed().unwrap().as_secs_f32()
        );
        eprintln!("---------");
    }

    eqs
}

fn all_lin_analysis_tot<'a, O: 'a + MonomialOrdering, C: Cipher + Clone + Sync>(
    cipher: &mut C,
    zz: &[Message],
    is_forward: bool,
    vars_l: &[Vec<Vec<Monomial<'a, O>>>],
    excluded_mons: &BTreeSet<Monomial<'a, O>>,
    print: bool,
    ring: &'a BoxedRing<O>,
) -> LinkedList<Polynomial<'a, O>> {
    let tot_time = SystemTime::now();
    let mut vars = Vec::new();
    for iter_nm in vars_l.iter() {
        for iter_nr in iter_nm.iter() {
            for v in iter_nr.iter() {
                if !excluded_mons.contains(v) {
                    vars.push(v.clone());
                }
            }
        }
    }
    vars.push(Monomial::one(ring));
    vars.sort();
    vars.reverse();
    let eqs = prone_polynomials::<O, C>(&vars, is_forward, cipher, zz, print, ring);

    if print {
        eprintln!("---------");
        eprintln!(
            "#eqs: {} , duration: {}",
            eqs.len(),
            tot_time.elapsed().unwrap().as_secs_f32()
        );
        eprintln!("---------");
    }
    eqs
}
