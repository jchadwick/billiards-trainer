//! Exact bounded K-best cyclic paths through observed radial transitions.
//! A minimum-cost path is an image hypothesis, never proof of a silhouette.
use super::*;

#[derive(Clone, Debug)]
pub(super) struct Path {
    pub cost: f64,
    pub indices: Vec<usize>,
}
pub(super) fn paths(
    candidates: &[Candidate],
    cfg: &BallLocalizationConfig,
    end: Instant,
) -> Result<Vec<Path>, RejectionReason> {
    let rank = |a: &Path, b: &Path| {
        a.cost.total_cmp(&b.cost).then_with(|| {
            a.indices
                .iter()
                .zip(&b.indices)
                .find_map(|(&i, &j)| {
                    let order = candidates[i].point[0]
                        .total_cmp(&candidates[j].point[0])
                        .then_with(|| candidates[i].point[1].total_cmp(&candidates[j].point[1]));
                    (order != std::cmp::Ordering::Equal).then_some(order)
                })
                .unwrap_or_else(|| a.indices.len().cmp(&b.indices.len()))
        })
    };
    let mut rays = vec![Vec::<usize>::new(); cfg.angular_samples];
    for (i, c) in candidates.iter().enumerate() {
        rays[c.ray].push(i);
    }
    for ray in &mut rays {
        ray.sort_by(|&i, &j| {
            distance(candidates[i].point, [0.; 2])
                .total_cmp(&distance(candidates[j].point, [0.; 2]))
                .then_with(|| candidates[i].point[0].total_cmp(&candidates[j].point[0]))
                .then_with(|| candidates[i].point[1].total_cmp(&candidates[j].point[1]))
        });
    }
    let active: Vec<_> = rays
        .iter()
        .enumerate()
        .filter(|(_, r)| !r.is_empty())
        .map(|(i, _)| i)
        .collect();
    if active.len() < cfg.minimum_inliers {
        return Err(RejectionReason::InsufficientBoundary);
    }
    let mut radii: Vec<_> = candidates
        .iter()
        .map(|c| distance(c.point, [0.; 2]))
        .collect();
    let scale = median(&mut radii).max(cfg.numerical_epsilon);
    let edge = |i: usize, j: usize, gap: usize| {
        let dr = distance(candidates[i].point, [0.; 2]) - distance(candidates[j].point, [0.; 2]);
        cfg.cyclic_continuity_weight * dr * dr
            / (scale * scale * std::f64::consts::TAU * gap as f64 / cfg.angular_samples as f64)
    };
    let node = |i: usize| -candidates[i].strength.max(cfg.numerical_epsilon).ln();
    let mut complete = Vec::new();
    for &start in &rays[active[0]] {
        let mut previous = vec![vec![Path {
            cost: node(start),
            indices: vec![start],
        }]];
        for layer in 1..active.len() {
            deadline(Instant::now(), end)?;
            let mut next = Vec::with_capacity(rays[active[layer]].len());
            for &index in &rays[active[layer]] {
                let mut proposals = Vec::new();
                for prefix in previous.iter().flatten() {
                    let mut value = prefix.clone();
                    value.cost += node(index)
                        + edge(
                            *prefix.indices.last().unwrap(),
                            index,
                            active[layer] - active[layer - 1],
                        );
                    value.indices.push(index);
                    proposals.push(value);
                }
                proposals.sort_by(rank);
                proposals.truncate(cfg.maximum_cyclic_paths);
                next.push(proposals);
            }
            previous = next;
        }
        for mut value in previous.into_iter().flatten() {
            value.cost += edge(
                *value.indices.last().unwrap(),
                start,
                cfg.angular_samples + active[0] - active[active.len() - 1],
            );
            complete.push(value);
        }
    }
    complete.sort_by(rank);
    complete.truncate(cfg.maximum_cyclic_paths);
    Ok(complete)
}

pub(super) fn initialize(points: &[[f64; 2]], eps: f64) -> Option<Circle> {
    let n = points.len() as f64;
    let origin = [
        points.iter().map(|p| p[0]).sum::<f64>() / n,
        points.iter().map(|p| p[1]).sum::<f64>() / n,
    ];
    let mut matrix = [[0.; 3]; 3];
    let mut rhs = [0.; 3];
    for p in points {
        let x = p[0] - origin[0];
        let y = p[1] - origin[1];
        let row = [2. * x, 2. * y, 1.];
        let value = x * x + y * y;
        for i in 0..3 {
            rhs[i] += row[i] * value;
            for j in 0..3 {
                matrix[i][j] += row[i] * row[j];
            }
        }
    }
    let v = solve(matrix, rhs, eps)?;
    let r2 = v[2] + v[0] * v[0] + v[1] * v[1];
    (r2 > 0. && r2.is_finite()).then_some(Circle {
        center: [v[0] + origin[0], v[1] + origin[1]],
        radius: r2.sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rings(count: usize) -> Vec<Candidate> {
        (0..count)
            .flat_map(|ray| {
                let a = std::f64::consts::TAU * ray as f64 / count as f64;
                [10., 13.].map(move |r| Candidate {
                    ray,
                    point: [r * a.cos(), r * a.sin()],
                    strength: 0.8,
                })
            })
            .collect()
    }
    fn config(count: usize) -> BallLocalizationConfig {
        BallLocalizationConfig {
            angular_samples: count,
            minimum_inliers: 3,
            maximum_cyclic_paths: 8,
            ..Default::default()
        }
    }
    #[test]
    fn closure_keeps_both_coherent_rings_and_input_order_invariance() {
        let c = rings(12);
        let cfg = config(12);
        let end = Instant::now() + Duration::from_secs(5);
        let a = paths(&c, &cfg, end).unwrap();
        assert!((a[0].cost - a[1].cost).abs() < 1e-10);
        let radii: Vec<_> = a[..2].iter().map(|p| c[p.indices[0]].point[0]).collect();
        assert!(radii.contains(&10.) && radii.contains(&13.));
        let reversed: Vec<_> = c.iter().rev().copied().collect();
        let b = paths(&reversed, &cfg, end).unwrap();
        let geometry =
            |p: &Path, cs: &[Candidate]| p.indices.iter().map(|&i| cs[i].point).collect::<Vec<_>>();
        assert_eq!(
            a.iter().map(|p| geometry(p, &c)).collect::<Vec<_>>(),
            b.iter().map(|p| geometry(p, &reversed)).collect::<Vec<_>>()
        );
    }
    #[test]
    fn exact_top_paths_match_exhaustive_four_ray_cycles() {
        let c = rings(4);
        let cfg = config(4);
        let result = paths(&c, &cfg, Instant::now() + Duration::from_secs(5)).unwrap();
        let scale = 11.5;
        let mut costs = Vec::new();
        for mask in 0..16 {
            let radii: Vec<f64> = (0..4)
                .map(|r| if mask & (1 << r) == 0 { 10. } else { 13. })
                .collect();
            let cost = -4. * 0.8_f64.ln()
                + (0..4)
                    .map(|i| {
                        (radii[i] - radii[(i + 1) % 4]).powi(2)
                            / (scale * scale * std::f64::consts::FRAC_PI_2)
                    })
                    .sum::<f64>();
            costs.push(cost);
        }
        costs.sort_by(f64::total_cmp);
        for (a, b) in result.iter().zip(costs) {
            assert!((a.cost - b).abs() < 1e-10);
        }
    }
    #[test]
    fn known_offset_circle_initialization_and_deadline() {
        let pts: Vec<_> = (0..96)
            .map(|i| {
                let a = std::f64::consts::TAU * i as f64 / 96.;
                [2. + 13. * a.cos(), -3. + 13. * a.sin()]
            })
            .collect();
        let fit = initialize(&pts, 1e-9).unwrap();
        assert!(distance(fit.center, [2., -3.]) < 1e-9);
        assert!((fit.radius - 13.).abs() < 1e-9);
        assert!(matches!(
            paths(&rings(12), &config(12), Instant::now()),
            Err(RejectionReason::DeadlineExceeded)
        ));
    }
}
