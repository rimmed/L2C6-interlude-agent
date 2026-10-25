/*
 * Copyright (c) 2013 L2jMobius
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
 * IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.l2jmobius.gameserver.managers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import org.l2jmobius.commons.util.Rnd;
import org.l2jmobius.gameserver.config.custom.RLAgentEnvironmentConfig;
import org.l2jmobius.gameserver.data.SpawnTable;
import org.l2jmobius.gameserver.data.xml.SpawnData;
import org.l2jmobius.gameserver.geoengine.GeoEngine;
import org.l2jmobius.gameserver.model.Location;
import org.l2jmobius.gameserver.model.World;
import org.l2jmobius.gameserver.model.actor.Npc;
import org.l2jmobius.gameserver.model.actor.Player;
import org.l2jmobius.gameserver.model.actor.enums.creature.Race;
import org.l2jmobius.gameserver.model.actor.instance.Monster;
import org.l2jmobius.gameserver.model.spawns.Spawn;

/**
 * Farm areas of the RL agents, built from the server's own monster spawns.<br>
 * An area is a spawn territory (or, for point spawns, a grid cell) of ordinary monsters. An area suits an agent of level L when no monster is above {@code L + RLFarmLevelAbove} and most monsters are at least {@code L - RLFarmLevelBelow}.<br>
 * Undead hunters (support casters, {@link RLAgentManager.Build#UNDEAD_MAGE}) prefer areas where undead are the majority, with a wider window below their level ({@code RLFarmUndeadLevelBelow}).<br>
 * Every agent is assigned to one area at a time, and an area takes at most one agent per {@code RLFarmMobsPerAgent} fitting monsters, so hundreds of agents spread over the map instead of competing for the same monsters.
 */
final class RLFarmAreas
{
	private static final Logger LOGGER = Logger.getLogger(RLFarmAreas.class.getName());
	
	/** Size of the grid cell that groups point spawns (spawns without a territory). */
	private static final int CELL_SIZE = 2000;
	/** Random offset around the chosen monster, so the agent does not land inside it. */
	private static final int MONSTER_OFFSET = 100;
	/** Radius checked around a spot for monsters above the level window (areas of different levels can touch) and aggressive ones. */
	private static final int SAFETY_RADIUS = 1500;
	/** Extra distance beyond an aggressive monster's aggro range. */
	private static final int AGGRO_MARGIN = 150;
	/** Live monsters of the area tried as the spot before the last one is taken anyway. */
	private static final int SPOT_ATTEMPTS = 12;
	
	/**
	 * Monsters of one area.
	 * @param key territory name or grid cell
	 * @param spawns the monster spawns of the area
	 * @param amounts monster count per level
	 * @param undeadAmounts undead monster count per level
	 * @param maxLevel highest monster level
	 */
	private record Area(String key, List<Spawn> spawns, Map<Integer, Integer> amounts, Map<Integer, Integer> undeadAmounts, int maxLevel)
	{
		/** Undead monsters fitting an undead hunter of {@code level} (a wider window below), 0 when the area does not suit it. */
		int fittingUndead(int level)
		{
			if (maxLevel > (level + RLAgentEnvironmentConfig.RL_FARM_LEVEL_ABOVE))
			{
				return 0;
			}
			
			final int lowest = level - RLAgentEnvironmentConfig.RL_FARM_UNDEAD_LEVEL_BELOW;
			int inWindow = 0;
			for (Map.Entry<Integer, Integer> entry : amounts.entrySet())
			{
				if (entry.getKey() >= lowest)
				{
					inWindow += entry.getValue();
				}
			}
			int undead = 0;
			for (Map.Entry<Integer, Integer> entry : undeadAmounts.entrySet())
			{
				if (entry.getKey() >= lowest)
				{
					undead += entry.getValue();
				}
			}
			return ((undead >= RLAgentEnvironmentConfig.RL_FARM_MIN_MONSTERS) && ((undead * 100) >= (inWindow * RLAgentEnvironmentConfig.RL_FARM_UNDEAD_MIN_SHARE))) ? undead : 0;
		}
		
		int fitting(int level, boolean undead)
		{
			return undead ? fittingUndead(level) : fittingMonsters(level);
		}

		/** Monsters of the area within the level window of {@code level}, 0 when the area does not suit that level. */
		int fittingMonsters(int level)
		{
			if (maxLevel > (level + RLAgentEnvironmentConfig.RL_FARM_LEVEL_ABOVE))
			{
				return 0;
			}
			
			int total = 0;
			int fitting = 0;
			for (Map.Entry<Integer, Integer> entry : amounts.entrySet())
			{
				total += entry.getValue();
				if (entry.getKey() >= (level - RLAgentEnvironmentConfig.RL_FARM_LEVEL_BELOW))
				{
					fitting += entry.getValue();
				}
			}
			if ((fitting < RLAgentEnvironmentConfig.RL_FARM_MIN_MONSTERS) || ((fitting * 100) < (total * RLAgentEnvironmentConfig.RL_FARM_MIN_SHARE)))
			{
				return 0;
			}
			return fitting;
		}
		
		int capacity(int level, boolean undead)
		{
			return Math.max(1, fitting(level, undead) / RLAgentEnvironmentConfig.RL_FARM_MOBS_PER_AGENT);
		}
	}
	
	private volatile List<Area> _areas;
	private final Map<String, Area> _byKey = new HashMap<>();
	/** Agent object id -> key of its area. */
	private final Map<Integer, String> _assignments = new ConcurrentHashMap<>();
	
	/**
	 * Picks the farm spot of an agent: its current area while that still suits its level, otherwise the least crowded suitable area.
	 * @param player the agent
	 * @return a spot next to a monster of the area, or {@code null} when no area suits the agent's level
	 */
	synchronized Location pickSpot(Player player)
	{
		final int level = player.getLevel();
		// Undead hunters (support casters) go where undead live; without such an area, anywhere that fits.
		final boolean undead = RLAgentManager.buildOf(player) == RLAgentManager.Build.UNDEAD_MAGE;
		Area area = null;
		final String current = _assignments.get(player.getObjectId());
		if (current != null)
		{
			final Area assigned = _byKey.get(current);
			if ((assigned != null) && ((undead && (assigned.fittingUndead(level) > 0)) || (!undead && (assigned.fittingMonsters(level) > 0))))
			{
				area = assigned;
			}
		}
		
		if ((area == null) && undead)
		{
			area = leastCrowded(level, player.getObjectId(), true, null);
		}
		if (area == null)
		{
			area = leastCrowded(level, player.getObjectId(), false, null);
		}
		if (area == null)
		{
			_assignments.remove(player.getObjectId());
			return null;
		}
		// Recorded for undead hunters too: the crowd of an area counts every agent sent there.
		_assignments.put(player.getObjectId(), area.key());
		return spotIn(area, level);
	}
	
	/**
	 * @param player the agent
	 * @return {@code true} when the agent's area no longer suits its level (e.g. after a level-up)
	 */
	boolean outgrown(Player player)
	{
		areas();
		final String key = _assignments.get(player.getObjectId());
		final Area area = key == null ? null : _byKey.get(key);
		final boolean undead = RLAgentManager.buildOf(player) == RLAgentManager.Build.UNDEAD_MAGE;
		return (area == null) || ((area.fittingMonsters(player.getLevel()) == 0) && (!undead || (area.fittingUndead(player.getLevel()) == 0)));
	}
	
	/**
	 * A spot in another area than the agent's current one (an agent reset to its base level after a level-up moves on, so it sees new places).
	 * @param player the agent
	 * @return the spot, or {@code null} when no other area suits its level
	 */
	synchronized Location relocate(Player player)
	{
		final int level = player.getLevel();
		final boolean undead = RLAgentManager.buildOf(player) == RLAgentManager.Build.UNDEAD_MAGE;
		final String current = _assignments.get(player.getObjectId());
		Area area = undead ? leastCrowded(level, player.getObjectId(), true, current) : null;
		if (area == null)
		{
			area = leastCrowded(level, player.getObjectId(), false, current);
		}
		if (area == null)
		{
			return null;
		}
		_assignments.put(player.getObjectId(), area.key());
		return spotIn(area, level);
	}
	
	void release(Player player)
	{
		_assignments.remove(player.getObjectId());
	}
	
	private Area leastCrowded(int level, int objectId, boolean undead, String excluded)
	{
		final Map<String, Integer> crowd = new HashMap<>();
		for (Map.Entry<Integer, String> entry : _assignments.entrySet())
		{
			if (entry.getKey() != objectId)
			{
				crowd.merge(entry.getValue(), 1, Integer::sum);
			}
		}
		
		// Weighted by free places; when every suitable area is full, the least loaded one.
		final List<Area> free = new ArrayList<>();
		final List<Integer> weights = new ArrayList<>();
		Area leastLoaded = null;
		double lowestLoad = Double.MAX_VALUE;
		int totalWeight = 0;
		for (Area area : areas())
		{
			if ((area.fitting(level, undead) == 0) || area.key().equals(excluded))
			{
				continue;
			}
			
			final int capacity = area.capacity(level, undead);
			final int agents = crowd.getOrDefault(area.key(), 0);
			if (agents < capacity)
			{
				free.add(area);
				weights.add(capacity - agents);
				totalWeight += capacity - agents;
			}
			final double load = (double) agents / capacity;
			if (load < lowestLoad)
			{
				lowestLoad = load;
				leastLoaded = area;
			}
		}
		
		if (free.isEmpty())
		{
			return leastLoaded;
		}
		
		int pick = Rnd.get(totalWeight);
		for (int i = 0; i < free.size(); i++)
		{
			pick -= weights.get(i);
			if (pick < 0)
			{
				return free.get(i);
			}
		}
		return free.get(free.size() - 1);
	}
	
	private static Location spotIn(Area area, int level)
	{
		// Next to a live monster: right height, and there is something to fight. Dead areas fall back to a spawn point.
		final List<Npc> alive = new ArrayList<>();
		for (Spawn spawn : area.spawns())
		{
			for (Npc npc : spawn.getSpawnedNpcs())
			{
				if (!npc.isDead() && npc.isSpawned())
				{
					alive.add(npc);
				}
			}
		}
		
		Location spot = null;
		Collections.shuffle(alive);
		for (int i = 0; (i < alive.size()) && (i < SPOT_ATTEMPTS); i++)
		{
			spot = spotNear(alive.get(i));
			if (isSafe(alive.get(i), spot, level))
			{
				return spot;
			}
		}
		if (spot != null)
		{
			return spot; // no safe spot among the tried monsters: the last one
		}
		
		// A random spawn point; some territories give none, so try a few spawns before giving up.
		for (int attempt = 0; attempt < SPOT_ATTEMPTS; attempt++)
		{
			final Spawn spawn = area.spawns().get(Rnd.get(area.spawns().size()));
			final Location base = spawn.isTerritoryBased() ? spawn.getSpawnTerritory().getRandomPoint() : spawn.getSpawnLocation();
			if (base != null)
			{
				return onGround(base, base.getX(), base.getY());
			}
		}
		return null; // the caller falls back to the farm zone anchor
	}
	
	/** A point next to the monster; next to an aggressive one, just outside its aggro range so it does not attack at once. */
	private static Location spotNear(Npc npc)
	{
		// A random point next to the monster that it can walk to: a point behind a wall (Cemetery) would put the agent on the other side.
		final Location base = npc.getLocation();
		for (int attempt = 0; attempt < SPOT_ATTEMPTS; attempt++)
		{
			final Location spot;
			if (!npc.isAggressive())
			{
				spot = onGround(base, base.getX() + Rnd.get(-MONSTER_OFFSET, MONSTER_OFFSET), base.getY() + Rnd.get(-MONSTER_OFFSET, MONSTER_OFFSET));
			}
			else
			{
				final double angle = Rnd.nextDouble() * 2 * Math.PI;
				final int distance = npc.getAggroRange() + AGGRO_MARGIN;
				spot = onGround(base, base.getX() + (int) (Math.cos(angle) * distance), base.getY() + (int) (Math.sin(angle) * distance));
			}
			if (GeoEngine.getInstance().canMoveToTarget(base, spot))
			{
				return spot;
			}
		}
		return base;
	}
	
	/** A spot is unsafe when a monster above the level window is near, or an aggressive monster would reach it at once. */
	private static boolean isSafe(Npc anchor, Location spot, int level)
	{
		for (Monster monster : World.getInstance().getVisibleObjectsInRange(anchor, Monster.class, SAFETY_RADIUS))
		{
			if (monster.isDead() || !monster.getTemplate().isType("Monster"))
			{
				continue;
			}
			if (monster.getLevel() > (level + RLAgentEnvironmentConfig.RL_FARM_LEVEL_ABOVE))
			{
				return false;
			}
			if (monster.isAggressive() && (Math.hypot(monster.getX() - spot.getX(), monster.getY() - spot.getY()) < monster.getAggroRange()))
			{
				return false;
			}
		}
		return true;
	}
	
	/** {@code (x, y)} on the ground near {@code base}, or {@code base} itself when the ground there is far off (a cliff, a wall). */
	private static Location onGround(Location base, int x, int y)
	{
		final int z = GeoEngine.getInstance().getHeight(x, y, base.getZ());
		return Math.abs(z - base.getZ()) > 200 ? new Location(base.getX(), base.getY(), base.getZ()) : new Location(x, y, z);
	}
	
	private List<Area> areas()
	{
		List<Area> areas = _areas;
		if (areas == null)
		{
			synchronized (this)
			{
				areas = _areas;
				if (areas == null)
				{
					areas = build();
					for (Area area : areas)
					{
						_byKey.put(area.key(), area);
					}
					_areas = areas;
				}
			}
		}
		return areas;
	}
	
	private static List<Area> build()
	{
		final Map<String, List<Spawn>> grouped = new LinkedHashMap<>();
		int skipped = 0;
		for (Set<Spawn> spawns : SpawnTable.getInstance().getSpawnTable().values())
		{
			for (Spawn spawn : spawns)
			{
				if (!spawn.getTemplate().isType("Monster") || isExcluded(spawn) || RLAgentEnvironmentConfig.RL_FARM_EXCLUDED_MONSTERS.contains(spawn.getTemplate().getId()))
				{
					skipped++;
					continue;
				}
				
				final String key = spawn.isTerritoryBased() ? "T:" + spawn.getSpawnTerritory().getName() : "C:" + Math.floorDiv(spawn.getX(), CELL_SIZE) + "," + Math.floorDiv(spawn.getY(), CELL_SIZE);
				grouped.computeIfAbsent(key, k -> new ArrayList<>()).add(spawn);
			}
		}
		
		final List<Area> areas = new ArrayList<>();
		for (Map.Entry<String, List<Spawn>> entry : grouped.entrySet())
		{
			final Map<Integer, Integer> amounts = new HashMap<>();
			final Map<Integer, Integer> undeadAmounts = new HashMap<>();
			int maxLevel = 0;
			for (Spawn spawn : entry.getValue())
			{
				final int level = spawn.getTemplate().getLevel();
				amounts.merge(level, Math.max(1, spawn.getAmount()), Integer::sum);
				if (spawn.getTemplate().getRace() == Race.UNDEAD)
				{
					undeadAmounts.merge(level, Math.max(1, spawn.getAmount()), Integer::sum);
				}
				maxLevel = Math.max(maxLevel, level);
			}
			areas.add(new Area(entry.getKey(), List.copyOf(entry.getValue()), Map.copyOf(amounts), Map.copyOf(undeadAmounts), maxLevel));
		}
		
		LOGGER.info(RLFarmAreas.class.getSimpleName() + ": " + areas.size() + " farm areas from monster spawns (" + skipped + " other spawns ignored), level window -" + RLAgentEnvironmentConfig.RL_FARM_LEVEL_BELOW + "/+" + RLAgentEnvironmentConfig.RL_FARM_LEVEL_ABOVE + ", " + RLAgentEnvironmentConfig.RL_FARM_MOBS_PER_AGENT + " monsters per agent.");
		return Collections.unmodifiableList(areas);
	}
	
	private static boolean isExcluded(Spawn spawn)
	{
		final String file = SpawnData.getInstance().getSpawnFile(spawn.getNpcSpawnTemplateId());
		if (file == null)
		{
			return false;
		}
		
		final String path = file.replace('\\', '/');
		for (String folder : RLAgentEnvironmentConfig.RL_FARM_EXCLUDED_SPAWN_FOLDERS)
		{
			// a folder (data/spawns/<folder>/) or a single spawn file (<name>.xml)
			if (path.contains("/" + folder + "/") || path.endsWith("/" + folder + ".xml"))
			{
				return true;
			}
		}
		return false;
	}
}
